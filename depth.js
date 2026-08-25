// depth.js — DINOv2-based monocular depth estimation, with a zero-download
// fallback so the demo always runs.
//
// The original clip uses Meta FAIR's DINOv2 to estimate a per-pixel depth map.
// Depth-Anything reuses a DINOv2 (ViT) encoder as its backbone, so we run
// `depth-anything-small` in the browser via @huggingface/transformers
// (ONNX Runtime Web, WebGPU backend) when available.
//
// Two paths:
//   • REAL: transformers.js loads the ONNX model (from the Hub, or locally).
//   • SYNTHETIC: if the model can't be fetched (blocked network / not present),
//     we approximate depth from frame luminance + a radial bias. No download,
//     works instantly, and still drives the lighting/shadow engine.

import { pipeline, env } from '@huggingface/transformers';

env.backends.onnx.wasm.proxy = false;

// ---- model selection -------------------------------------------------------
// Set USE_LOCAL = true after you drop the model files into
//   hand-light-depth/models/depth-anything-small-hf/
// (see README for the exact file list). Local loading needs no internet.
// Even with USE_LOCAL=true, remote sources are kept as a fallback, so a
// missing or mis-named local folder won't hard-fail the whole load.
const USE_LOCAL = true;
const LOCAL_MODEL_ID = 'depth-anything-small-hf'; // folder under localModelPath
// Candidate remote sources, tried in order. Add/remove as needed.
const REMOTE_MODEL_IDS = [
  'onnx-community/depth-anything-small-hf',
  'Xenova/depth-anything-small-hf',
];
// Force the depth model onto WebGPU. Your RTX 4060 (8GB) is more than enough.
// The earlier silent fallback to WASM was onnxruntime-web failing its own
// requestAdapter() on Windows (the "powerPreference ignored" warning); we
// work around it by handing onnxruntime-web a GPUDevice we acquire ourselves.
// Set this to false to allow the WASM fallback again.
const FORCE_WEBGPU = true;
// For each source we try WebGPU first (now made reliable via the shared
// GPUDevice below), then fall back to WASM (CPU) only if WebGPU truly can't
// run. WASM still runs the real model, just slower.
const DEVICE_FALLBACKS = [
  { device: 'webgpu', dtype: 'fp32' },
  { device: 'wasm', dtype: 'fp32' },
];

// Local-first: prefer the model files in ./models/ when present, but keep the
// remote allowed so a missing/typo'd folder transparently falls back to the
// canonical remote source instead of hard-failing with a 404.
if (USE_LOCAL) {
  env.allowLocalModels = true;
  env.localModelPath = './models/';
}

export const DEPTH_W = 448;
export const DEPTH_H = 336;

let pipe = null;
let busy = false;
let loadStarted = false;
let synthCanvas = null;

// ---- synthetic depth (no download) -----------------------------------------
let synthCtx = null;
let synthRadial = null; // cached radial bias (depends only on W/H)
function syntheticDepth(source) {
  const W = 256, H = 192;
  if (!synthCanvas) {
    synthCanvas = document.createElement('canvas');
    // willReadFrequently avoids the per-frame "getImageData is faster with
    // willReadFrequently" warning and makes readback cheaper on 2D canvases.
    synthCtx = synthCanvas.getContext('2d', { willReadFrequently: true });
    // radial term is image-independent, precompute once onto a lookup array.
    synthRadial = new Float32Array(W * H);
    const cx = W / 2, cy = H / 2, maxR = Math.hypot(cx, cy);
    for (let y = 0; y < H; y++) {
      for (let x = 0; x < W; x++) {
        synthRadial[y * W + x] = 0.55 * (1 - Math.hypot(x - cx, y - cy) / maxR);
      }
    }
  }
  const c = synthCanvas;
  c.width = W; c.height = H;
  const ctx = synthCtx;
  ctx.drawImage(source, 0, 0, W, H);
  const img = ctx.getImageData(0, 0, W, H);
  const d = img.data;
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const i = (y * W + x) * 4;
      const luma = (0.299 * d[i] + 0.587 * d[i + 1] + 0.114 * d[i + 2]) / 255;
      let v = synthRadial[y * W + x] + 0.45 * (1 - luma); // center closer, bright = far
      v = v < 0 ? 0 : v > 1 ? 1 : v;
      const g = (v * 255) | 0;
      d[i] = d[i + 1] = d[i + 2] = g;
      d[i + 3] = 255;
    }
  }
  ctx.putImageData(img, 0, 0);
  return c;
}

// ---- WebGPU device sharing -------------------------------------------------
// On Windows, onnxruntime-web's own requestAdapter() can fail (the
// "powerPreference ignored" warning seen in console), silently dropping us to
// WASM even though the GPU is fine. We acquire a GPUDevice ourselves and hand
// it to onnxruntime-web so the WebGPU execution provider is actually used.
// transformers.js initializes its onnx backend lazily, so we re-apply the
// device right before each WebGPU pipeline attempt (idempotent + cached).
let sharedGpuDevice = null;
let sharedGpuTried = false;
async function ensureWebGpuDevice() {
  if (!sharedGpuTried) {
    sharedGpuTried = true;
    if (typeof navigator !== 'undefined' && navigator.gpu) {
      try {
        const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
        if (adapter) {
          sharedGpuDevice = await adapter.requestDevice();
          console.log('[depth] acquired WebGPU device for onnxruntime-web');
        }
      } catch (e) {
        console.warn('[depth] WebGPU device acquisition failed:', e);
      }
    }
  }
  // apply (or re-apply once the onnx backend env is ready)
  if (sharedGpuDevice && env.backends?.onnx?.webgpu) {
    env.backends.onnx.webgpu.device = sharedGpuDevice;
    env.backends.onnx.webgpu.powerPreference = 'high-performance';
  }
  return sharedGpuDevice;
}

// ---- real model loading -----------------------------------------------------
export async function loadDepthModel(onStatus = () => {}) {
  if (loadStarted) return;
  loadStarted = true;
  // The renderer can use synthetic depth immediately; try the real model in
  // the background and switch over automatically when it's ready.
  onStatus('亮度近似深度已就绪（无需下载）；后台尝试加载真实 DINOv2…');
  (async () => {
    await ensureWebGpuDevice();
    const ids = USE_LOCAL ? [LOCAL_MODEL_ID, ...REMOTE_MODEL_IDS] : REMOTE_MODEL_IDS;
    let lastErr = null;
    for (const id of ids) {
      for (const cfg of DEVICE_FALLBACKS) {
        try {
          if (cfg.device === 'webgpu') await ensureWebGpuDevice();
          onStatus(`下载 DINOv2: ${id} (${cfg.device}) …`);
          pipe = await pipeline('depth-estimation', id, {
            ...cfg,
            progress_callback: (p) => {
              if (p.status === 'progress') {
                onStatus(`下载 DINOv2 ${id} (${cfg.device}): ${Math.round(p.progress ?? 0)}%`);
              }
            },
          });
          onStatus(`✅ DINOv2 真实深度已启用（${id} · ${cfg.device}，比近似更准）`);
          return;
        } catch (e) {
          lastErr = e;
          console.error(`depth model failed: ${id} (${cfg.device})`, e);
          onStatus(`⚠️ ${id} (${cfg.device}) 失败，尝试下一个…`);
        }
      }
    }
    onStatus('⚠️ 所有 DINOv2 源均加载失败（网络/文件缺失）：' +
             (lastErr ? lastErr.message : '') + ' — 继续使用亮度近似深度');
    if (lastErr) console.error(lastErr);
  })();
  return Promise.resolve();
}

// Convert the pipeline's predicted_depth tensor to a grayscale canvas.
function tensorToDepthCanvas(tensor) {
  const data = tensor.data;
  const dims = tensor.dims;
  let h, w;
  if (dims.length === 3) [, h, w] = dims;
  else [h, w] = dims;
  let min = Infinity, max = -Infinity;
  for (let i = 0; i < data.length; i++) {
    const v = data[i];
    if (v < min) min = v;
    if (v > max) max = v;
  }
  const range = max - min || 1;
  const canvas = document.createElement('canvas');
  canvas.width = w; canvas.height = h;
  const ctx = canvas.getContext('2d');
  const img = ctx.createImageData(w, h);
  for (let i = 0; i < data.length; i++) {
    const n = (data[i] - min) / range;
    const g = Math.round(n * 255);
    img.data[i * 4] = g;
    img.data[i * 4 + 1] = g;
    img.data[i * 4 + 2] = g;
    img.data[i * 4 + 3] = 255;
  }
  ctx.putImageData(img, 0, 0);
  return canvas;
}

// Estimate depth for `source` (canvas/video) and call onResult(canvas, ms).
export async function estimateDepth(source, onResult, onStatus = () => {}) {
  if (pipe) {
    if (busy) return;
    busy = true;
    const t0 = performance.now();
    try {
      const out = await pipe(source);
      onResult(tensorToDepthCanvas(out.predicted_depth), performance.now() - t0);
    } catch (e) {
      onStatus('深度推理出错：' + e.message + '（本帧用近似深度）');
      console.error(e);
      onResult(syntheticDepth(source), 0);
    } finally {
      busy = false;
    }
  } else {
    onResult(syntheticDepth(source), 0);
  }
}
