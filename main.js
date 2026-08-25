// main.js — orchestrates webcam + DINOv2 depth + MediaPipe hands + the renderer.
//
// Two decoupled loops, like the original clip:
//   • renderLoop (rAF): fast path — read the latest hand/mouse light position,
//     copy the current camera frame + latest depth map into GPU textures, and
//     draw the lit result. This is the "深度+手势 < 25ms" part.
//   • depthLoop (async, off the rAF path): runs DINOv2 depth on a 448x336
//     downscale as fast as the model allows and updates the depth texture.
// Hand tracking (MediaPipe) runs inside the render loop because it is cheap.

import { loadDepthModel, estimateDepth, getDepthSize, setDepthResolution } from './depth.js';
import { loadHandModel, detectHands } from './hands.js';
import { GpuRenderer, Canvas2DRenderer } from './gpu.js';

const els = {
  start: document.getElementById('start'),
  status: document.getElementById('status'),
  fps: document.getElementById('fps'),
  canvas: document.getElementById('output'),
  video: document.getElementById('cam'),
  intensity: document.getElementById('intensity'),
  ambient: document.getElementById('ambient'),
  depthScale: document.getElementById('depthScale'),
  shadowSoft: document.getElementById('shadowSoft'),
  lightHeight: document.getElementById('lightHeight'),
  lightRadius: document.getElementById('lightRadius'),
  mirror: document.getElementById('mirror'),
  color: document.getElementById('color'),
  lightLabel: document.getElementById('lightLabel'),
  enableReal: document.getElementById('enableReal'),
  depthRes: document.getElementById('depthRes'),
  winSize: document.getElementById('winSize'),
  winSizeVal: document.getElementById('v-winSize'),
  orient: document.getElementById('orient'),
};

function sleep(ms) { return new Promise((r) => setTimeout(r, ms)); }

const state = {
  running: false,
  renderer: null,
  useWebGPU: true,
  depthCanvas: null, // latest depth map (grayscale)
  depthReady: false,
  depthMs: 0,
  handReady: false,
  mirror: false,
  mouse: { x: 0.5, y: 0.5, active: false },
  light: { x: 0.5, y: 0.5, intensity: 0.6 },
};

function setStatus(msg) { els.status.textContent = msg; }

// initial placeholder depth (flat) so the renderer has something to sample
function makePlaceholderDepth() {
  const { w, h } = getDepthSize();
  const c = document.createElement('canvas');
  c.width = w; c.height = h;
  const ctx = c.getContext('2d');
  ctx.fillStyle = '#808080';
  ctx.fillRect(0, 0, c.width, c.height);
  return c;
}

async function startCamera() {
  const stream = await navigator.mediaDevices.getUserMedia({
    video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: 'user' },
    audio: false,
  });
  els.video.srcObject = stream;
  await els.video.play();
  // wait for dimensions
  if (!els.video.videoWidth) {
    await new Promise((r) => { els.video.onloadedmetadata = r; });
  }
  // NOTE: the output canvas size is owned by layoutCanvas() (driven by the
  // window-size + orientation controls), not by the camera frame, so the
  // demo can be portrait/square regardless of the landscape webcam.
}

async function initRenderer() {
  try {
    if (!navigator.gpu) throw new Error('WebGPU unavailable');
    const r = new GpuRenderer(els.canvas);
    await r.init();
    state.renderer = r;
    state.useWebGPU = true;
    setStatus('WebGPU renderer ready (TypeGPU device).');
  } catch (e) {
    console.warn('Falling back to 2D canvas renderer:', e.message);
    state.renderer = new Canvas2DRenderer(els.canvas);
    state.useWebGPU = false;
    setStatus('2D fallback renderer (no WebGPU): ' + e.message);
  }
}

function getUniforms() {
  return {
    x: state.light.x,
    y: state.light.y,
    intensity: parseFloat(els.intensity.value),
    ambient: parseFloat(els.ambient.value),
    depthScale: parseFloat(els.depthScale.value),
    shadowSoft: parseFloat(els.shadowSoft.value),
    lightHeight: parseFloat(els.lightHeight.value),
    lightRadius: parseFloat(els.lightRadius.value),
    aspect: (els.canvas.width || 1280) / (els.canvas.height || 720),
    videoAspect: (els.video.videoWidth || 1280) / (els.video.videoHeight || 720),
    mirror: state.mirror ? 1 : 0,
    color: hexToRgb(els.color.value),
  };
}

// Map a coordinate in the raw (landscape) camera frame to the DISPLAY frame,
// undoing the renderer's "cover" crop so the hand light lands where the hand
// actually appears on screen — for any window aspect ratio.
function sourceToDisplay(sx, sy) {
  const a = (els.canvas.width / els.canvas.height) || 1;             // display aspect
  const b = (els.video.videoWidth / els.video.videoHeight) || (16 / 9); // video aspect
  const R = a / b;
  if (R > 1) return [sx, (sy - 0.5) * R + 0.5];
  return [(sx - 0.5) / R + 0.5, sy];
}

function hexToRgb(hex) {
  const n = parseInt(hex.slice(1), 16);
  return [((n >> 16) & 255) / 255, ((n >> 8) & 255) / 255, (n & 255) / 255];
}

// ---- depth inference loop (decoupled, runs as fast as the model allows) ----
const depthSrc = document.createElement('canvas');
const _ds = getDepthSize();
depthSrc.width = _ds.w; depthSrc.height = _ds.h;
const depthCtx = depthSrc.getContext('2d', { willReadFrequently: false });

// Throttle so the loop yields a real macrotask every iteration. Without the
// yield, the pipe===null branch's "await estimateDepth" only schedules a
// microtask, which starves the event loop: rAF never runs (black screen) and
// the page can't even process a refresh (the "frozen, can't reload" symptom).
const DEPTH_MIN_INTERVAL = 66; // ms (~15 fps cap on the depth update)
async function depthLoop() {
  while (state.running) {
    const t0 = performance.now();
    if (els.video.readyState >= 2 && els.video.videoWidth) {
      const { w, h } = getDepthSize();
      depthCtx.drawImage(els.video, 0, 0, w, h);
      await estimateDepth(depthSrc, (canvas, ms) => {
        state.depthCanvas = canvas;
        state.depthReady = true;
        state.depthMs = ms;
      }, setStatus);
    } else {
      await sleep(100);
    }
    // CRITICAL real yield — see note above.
    const dt = performance.now() - t0;
    if (dt < DEPTH_MIN_INTERVAL) await sleep(DEPTH_MIN_INTERVAL - dt);
  }
}

// ---- render loop (rAF): hand/mouse light + per-frame lighting ----
let lastFrame = performance.now();
let frames = 0;
let fpsT = performance.now();

function renderLoop() {
  if (!state.running) return;
  try {
  const now = performance.now();

  // 1) light position: hand if present, else mouse
  let target = null;
  if (state.handReady) {
    const h = detectHands(els.video, now);
    if (h.present) {
      const [dx, dy] = sourceToDisplay(h.x, h.y);
      const displayX = state.mirror ? (1 - dx) : dx;
      target = { x: displayX, y: dy, intensity: h.intensity };
      els.lightLabel.textContent = '💡 光源：手指 (DINOv2 + 跟手)';
    }
  }
  if (!target && state.mouse.active) {
    const displayX = state.mirror ? (1 - state.mouse.x) : state.mouse.x;
    target = { x: displayX, y: state.mouse.y, intensity: 0.7 };
    els.lightLabel.textContent = '💡 光源：鼠标';
  }
  if (!target) {
    target = { x: 0.5, y: 0.5, intensity: 0.55 };
    els.lightLabel.textContent = '💡 光源：默认 (把手放进画面，或移动鼠标)';
  }

  // smooth
  const k = 0.35;
  state.light.x += (target.x - state.light.x) * k;
  state.light.y += (target.y - state.light.y) * k;
  state.light.intensity += (target.intensity - state.light.intensity) * k;

  // 2) draw
  const depth = state.depthCanvas || makePlaceholderDepth();
  const u = getUniforms();
  u.x = state.light.x; u.y = state.light.y; u.intensity = state.light.intensity;
  state.renderer.renderFrame(els.video, depth, u);

  // 3) fps
  frames++;
  if (now - fpsT >= 500) {
    const fps = (frames * 1000 / (now - fpsT)).toFixed(0);
    const mode = state.useWebGPU ? 'WebGPU' : '2D';
    els.fps.textContent = `FPS ${fps} · 渲染 ${mode} · 深度 ${state.depthMs ? state.depthMs.toFixed(0) : '–'}ms/帧`;
    frames = 0; fpsT = now;
  }
  lastFrame = now;
  requestAnimationFrame(renderLoop);
  } catch (e) {
    state.running = false;
    setStatus('渲染出错: ' + e.message + '（详见 Console）');
    console.error('[renderLoop]', e);
  }
}

// ---- mouse fallback ----
els.canvas.addEventListener('mousemove', (e) => {
  const r = els.canvas.getBoundingClientRect();
  state.mouse.x = (e.clientX - r.left) / r.width;
  state.mouse.y = (e.clientY - r.top) / r.height;
  state.mouse.active = true;
});
els.canvas.addEventListener('mouseleave', () => { state.mouse.active = false; });

els.mirror.addEventListener('change', () => { state.mirror = els.mirror.checked; });

// opt-in: load the real DINOv2 model only when the user checks the box
// (e.g. after confirming the approximate path runs smoothly).
els.enableReal.addEventListener('change', () => {
  if (els.enableReal.checked) {
    loadDepthModel(setStatus).catch((e) => setStatus('Depth model error: ' + e.message));
  }
});

// ---- depth-inference resolution selector ----
// Changing resolution only resizes the depth source canvas + placeholder;
// the (already loaded) model just ingests a different-sized frame, so no
// reload is needed. Higher = crisper shadows but more GPU/CPU per frame.
els.depthRes.addEventListener('change', () => {
  const [w, h] = els.depthRes.value.split('x').map(Number);
  setDepthResolution(w, h);
  depthSrc.width = w; depthSrc.height = h;
  state.depthCanvas = makePlaceholderDepth();
  const tip = w >= 1280 ? '（高分辨率建议走 WebGPU）'
            : w <= 448 ? '（轻量，WASM/CPU 也流畅）' : '';
  setStatus(`深度推理分辨率：${w}×${h}${tip}`);
});

// ---- demo window layout: display width (slider) × orientation (aspect) ----
// Both the CSS display box AND the canvas backing store follow this, so the
// rendered image is never stretch-distorted. The renderer does a "cover" fit
// of the landscape webcam into whatever box you pick (portrait included).
function layoutCanvas() {
  const w = parseInt(els.winSize.value, 10);
  const aspect = parseFloat(els.orient.value); // e.g. 16/9, 9/16, 1
  const h = Math.round(w / aspect);
  els.canvas.style.width = w + 'px';
  els.canvas.style.height = h + 'px';
  // backing store: match display × DPR, clamped so we never allocate a huge
  // texture (keeps WebGPU/2D happy on every GPU). Aspect is preserved.
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  let iw = Math.round(w * dpr), ih = Math.round(h * dpr);
  const cap = Math.max(iw, ih);
  if (cap > 1920) { const s = 1920 / cap; iw = Math.round(iw * s); ih = Math.round(ih * s); }
  els.canvas.width = iw; els.canvas.height = ih;
  els.winSizeVal.textContent = `${w}×${h}px`;
}
els.winSize.addEventListener('input', layoutCanvas);
els.orient.addEventListener('change', layoutCanvas);
// Seed the width slider with the canvas's natural width (fills the stage),
// then apply the initial layout (landscape by default).
requestAnimationFrame(() => {
  const natural = Math.round(els.canvas.clientWidth) || 720;
  els.winSize.value = Math.max(320, Math.min(1600, natural));
  layoutCanvas();
});

async function start() {
  if (state.running) return;
  els.start.disabled = true;
  setStatus('Requesting camera…');
  try {
    await startCamera();
  } catch (e) {
    setStatus('Camera error: ' + e.message + ' (need https/localhost + permission)');
    els.start.disabled = false;
    return;
  }
  setStatus('Initializing renderer…');
  await initRenderer();

  // load hand model (lightweight, always on)
  loadHandModel(setStatus).then(() => {
    state.handReady = true;
    setStatus('Hand model ready — put your hand in frame.');
  }).catch((e) => setStatus('Hand model error: ' + e.message + ' (mouse fallback active)'));

  // load the heavy DINOv2 depth model ONLY if the user opted in (checkbox).
  // This keeps the default start path lightweight so the page never freezes.
  if (els.enableReal.checked) {
    loadDepthModel(setStatus).catch((e) => setStatus('Depth model error: ' + e.message));
  } else {
    setStatus('亮度近似深度已就绪（无需下载）；勾选「启用真实 DINOv2」可换更准的真模型。');
  }

  state.depthCanvas = makePlaceholderDepth();
  state.running = true;
  requestAnimationFrame(renderLoop);
  depthLoop();
}

els.start.addEventListener('click', start);

// live value readouts for the sliders
function bindVal(id, fmt) {
  const el = document.getElementById(id);
  const span = document.getElementById('v-' + id);
  if (!el || !span) return;
  const upd = () => { span.textContent = fmt ? fmt(el.value) : el.value; };
  el.addEventListener('input', upd);
  upd();
}
bindVal('intensity', (v) => (+v).toFixed(2));
bindVal('ambient', (v) => (+v).toFixed(2));
bindVal('depthScale', (v) => (+v).toFixed(1));
bindVal('shadowSoft', (v) => (+v).toFixed(1));
bindVal('lightHeight', (v) => (+v).toFixed(2));
bindVal('lightRadius', (v) => (+v).toFixed(3));
