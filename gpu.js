// gpu.js — real-time depth lighting + shadows, "cast back" onto the live scene.
//
// Rendering is a single full-screen pass:
//   1. sample the camera frame (the real scene)
//   2. sample the DINOv2 depth map, derive a surface normal from its gradient
//   3. a point light sits at the hand position (index fingertip) at some height
//   4. diffuse + specular from that light, distance attenuation,
//      plus a screen-space shadow march so objects occlude the light
//   5. composite the lit result back over the original frame (零拷贝: the
//      camera texture is sampled directly, no CPU readback)
//
// Built on TypeGPU (software-mansion/TypeGPU): we use `tgpu.init()` for the GPU
// runtime/device, and fall back to vanilla `navigator.gpu` if TypeGPU is
// unavailable. The pipeline + WGSL follow TypeGPU's "eject to vanilla WebGPU"
// philosophy so it runs without a bundler. A 2D-canvas fallback keeps the demo
// usable on browsers without WebGPU.

const WGSL = /* wgsl */ `
struct Uniforms {
  lightPos   : vec2f,  // 0
  intensity  : f32,    // 8
  ambient    : f32,    // 12
  depthScale : f32,    // 16
  shadowSoft : f32,    // 20
  aspect     : f32,    // 24
  lightHeight: f32,    // 28
  mirror     : f32,    // 32
  lightColor : vec3f,  // 48 (align 16)
  _pad       : f32,    // 60
};

@group(0) @binding(0) var samp     : sampler;
@group(0) @binding(1) var camTex   : texture_2d<f32>;
@group(0) @binding(2) var depthTex : texture_2d<f32>;
@group(0) @binding(3) var<uniform> U : Uniforms;

struct VSOut {
  @builtin(position) pos : vec4f,
  @location(0) uv : vec2f,
};

@vertex
fn vs(@builtin(vertex_index) vi : u32) -> VSOut {
  var p = array<vec2f, 3>(
    vec2f(-1.0, -1.0),
    vec2f( 3.0, -1.0),
    vec2f(-1.0,  3.0),
  );
  let xy = p[vi];
  var out : VSOut;
  out.pos = vec4f(xy, 0.0, 1.0);
  // map clip-space to UV with origin at top-left (matches video + MediaPipe)
  out.uv = vec2f((xy.x + 1.0) * 0.5, 1.0 - (xy.y + 1.0) * 0.5);
  return out;
}

fn sampleDepth(uv : vec2f) -> f32 {
  return textureSampleLevel(depthTex, samp, uv, 0.0).r;
}

@fragment
fn fs(in : VSOut) -> @location(0) vec4f {
  let m = select(0.0, 1.0, U.mirror > 0.5);
  let sx = in.uv.x * (1.0 - 2.0 * m) + m;            // mirrored sample x
  let lx = U.lightPos.x * (1.0 - 2.0 * m) + m;        // mirrored light x

  let color = textureSampleLevel(camTex, samp, vec2f(sx, in.uv.y), 0.0).rgb;

  let dim = textureDimensions(camTex);
  let texel = vec2f(1.0 / f32(dim.x), 1.0 / f32(dim.y));
  let dx = sampleDepth(vec2f(sx, in.uv.y) + vec2f(texel.x, 0.0))
         - sampleDepth(vec2f(sx, in.uv.y) - vec2f(texel.x, 0.0));
  let dy = sampleDepth(vec2f(sx, in.uv.y) + vec2f(0.0, texel.y))
         - sampleDepth(vec2f(sx, in.uv.y) - vec2f(0.0, texel.y));
  let normal = normalize(vec3f(-dx * U.depthScale, -dy * U.depthScale, 1.0));

  let frag  = vec2f(sx * U.aspect, in.uv.y);
  let light = vec2f(lx * U.aspect, U.lightPos.y);
  var toLight = vec3f(light - frag, U.lightHeight);
  let dist = length(toLight);
  let L = toLight / max(dist, 1e-4);

  // screen-space shadow march from fragment toward the light
  var occ = 0.0;
  let steps = 18;
  let baseD = sampleDepth(vec2f(sx, in.uv.y));
  for (var i = 1; i <= steps; i = i + 1) {
    let t = f32(i) / f32(steps);
    let suv = mix(vec2f(sx, in.uv.y), vec2f(lx, U.lightPos.y), t);
    let sd = sampleDepth(suv);
    let diff = baseD - sd;            // >0 => a closer surface sits between us and the light
    occ = occ + max(0.0, diff) * exp(-t * 3.0);
  }
  let shadow = clamp(1.0 - occ * U.shadowSoft, 0.0, 1.0);

  let diff = max(dot(normal, L), 0.0);
  let atten = 1.0 / (1.0 + 2.5 * dist * dist);
  let lit = U.ambient + diff * atten * U.intensity * shadow;

  var outc = color * lit * U.lightColor;
  let V = vec3f(0.0, 0.0, 1.0);
  let H = normalize(L + V);
  let spec = pow(max(dot(normal, H), 0.0), 32.0) * atten * U.intensity * shadow;
  outc = outc + vec3f(spec) * 0.35;

  return vec4f(clamp(outc, vec3f(0.0), vec3f(1.0)), 1.0);
}
`;

export class GpuRenderer {
  constructor(canvas) {
    this.canvas = canvas;
    this.device = null;
    this.root = null;
    this.usingTypeGPU = false;
    this.usingTgpuUniform = false;
    this.typedUniform = null;
    this.camTex = null;
    this.depthTex = null;
    this.bindGroup = null;
    this.camW = 0;
    this.camH = 0;
    this.depthW = 0;
    this.depthH = 0;
    this.uniformData = new Float32Array(16); // matches Uniforms layout
  }

  async init() {
    const canvas = this.canvas;
    // --- TypeGPU as the runtime foundation (with vanilla fallback) ---
    try {
      const mod = await import('typegpu');
      const tgpu = mod.default ?? mod;
      this.root = await tgpu.init();
      this.device = this.root.device;
      this.usingTypeGPU = true;
      console.log('[renderer] device from TypeGPU');
    } catch (e) {
      console.warn('[renderer] TypeGPU init failed, using vanilla WebGPU:', e.message);
      const adapter = await navigator.gpu.requestAdapter();
      if (!adapter) throw new Error('WebGPU not supported (no adapter).');
      this.device = await adapter.requestDevice();
    }

    const device = this.device;
    const context = canvas.getContext('webgpu');
    if (!context) throw new Error('Could not get webgpu context');
    const format = navigator.gpu.getPreferredCanvasFormat();
    context.configure({ device, format, alphaMode: 'opaque' });
    this.context = context;
    this.format = format;

    const module = device.createShaderModule({ code: WGSL });
    this.pipeline = device.createRenderPipeline({
      layout: 'auto',
      vertex: { module, entryPoint: 'vs' },
      fragment: { module, entryPoint: 'fs', targets: [{ format }] },
      primitive: { topology: 'triangle-list' },
    });

    this.sampler = device.createSampler({
      magFilter: 'linear',
      minFilter: 'linear',
      addressModeU: 'clamp-to-edge',
      addressModeV: 'clamp-to-edge',
    });

    // --- uniform buffer: vanilla GPUBuffer, packed manually to match the WGSL
    // Uniforms layout exactly (see the struct in WGSL above). Using TypeGPU's
    // typed buffer here risked a vec3f (lightColor) alignment mismatch that made
    // lightColor read as 0 -> the whole frame rendered black. TypeGPU is still
    // used for the device/runtime (verified working), per its "eject to vanilla"
    // philosophy, but the uniform is written as a raw Float32Array.
    this.uniformBuffer = device.createBuffer({
      size: this.uniformData.byteLength,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
  }

  _ensureTextures(camW, camH, depthW, depthH) {
    if (camW === this.camW && camH === this.camH &&
        depthW === this.depthW && depthH === this.depthH && this.bindGroup) {
      return;
    }
    const device = this.device;
    const mk = (w, h) => device.createTexture({
      size: [w, h, 1],
      format: 'rgba8unorm',
      usage: GPUTextureUsage.TEXTURE_BINDING |
             GPUTextureUsage.COPY_DST |
             GPUTextureUsage.RENDER_ATTACHMENT,
    });
    this.camTex = mk(camW, camH);
    this.depthTex = mk(depthW, depthH);
    this.camW = camW; this.camH = camH;
    this.depthW = depthW; this.depthH = depthH;
    this.bindGroup = device.createBindGroup({
      layout: this.pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: this.sampler },
        { binding: 1, resource: this.camTex.createView() },
        { binding: 2, resource: this.depthTex.createView() },
        { binding: 3, resource: { buffer: this.uniformBuffer } },
      ],
    });
  }

  // uniforms: { x, y, intensity, ambient, depthScale, shadowSoft, aspect,
  //             lightHeight, color:[r,g,b], mirror }
  renderFrame(video, depthCanvas, uniforms) {
    const device = this.device;
    const camW = video.videoWidth || 640;
    const camH = video.videoHeight || 480;
    const depthW = depthCanvas.width;
    const depthH = depthCanvas.height;
    this._ensureTextures(camW, camH, depthW, depthH);

    device.queue.copyExternalImageToTexture(
      { source: video },
      { texture: this.camTex },
      [camW, camH],
    );
    device.queue.copyExternalImageToTexture(
      { source: depthCanvas },
      { texture: this.depthTex },
      [depthW, depthH],
    );

    // pack uniforms
    const u = this.uniformData;
    u[0] = uniforms.x; u[1] = uniforms.y;
    u[2] = uniforms.intensity; u[3] = uniforms.ambient;
    u[4] = uniforms.depthScale; u[5] = uniforms.shadowSoft;
    u[6] = uniforms.aspect; u[7] = uniforms.lightHeight;
    u[8] = uniforms.mirror;
    u[12] = uniforms.color[0]; u[13] = uniforms.color[1]; u[14] = uniforms.color[2];
    // indices 9,10,11 and 15 are padding (zeros)

    device.queue.writeBuffer(this.uniformBuffer, 0, u);

    const encoder = device.createCommandEncoder();
    const pass = encoder.beginRenderPass({
      colorAttachments: [{
        view: this.context.getCurrentTexture().createView(),
        loadOp: 'clear',
        storeOp: 'store',
      }],
    });
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, this.bindGroup);
    pass.draw(3);
    pass.end();
    device.queue.submit([encoder.finish()]);
  }
}

// --- No-WebGPU fallback: approximate the effect on a 2D canvas. ---
export class Canvas2DRenderer {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
  }

  renderFrame(video, depthCanvas, uniforms) {
    const ctx = this.ctx;
    const w = video.videoWidth || 640;
    const h = video.videoHeight || 480;
    if (this.canvas.width !== w) this.canvas.width = w;
    if (this.canvas.height !== h) this.canvas.height = h;

    ctx.save();
    if (uniforms.mirror) {
      ctx.translate(w, 0);
      ctx.scale(-1, 1);
    }
    ctx.drawImage(video, 0, 0, w, h);
    ctx.restore();

    // additive light pool at the hand position
    const lx = uniforms.mirror ? (1 - uniforms.x) * w : uniforms.x * w;
    const ly = uniforms.y * h;
    const radius = Math.max(w, h) * 0.45;
    const g = ctx.createRadialGradient(lx, ly, 0, lx, ly, radius);
    const [r, gg, b] = uniforms.color;
    const a = uniforms.intensity * 0.6;
    g.addColorStop(0, `rgba(${(r*255)|0},${(gg*255)|0},${(b*255)|0},${a})`);
    g.addColorStop(1, 'rgba(0,0,0,0)');
    ctx.globalCompositeOperation = 'lighter';
    ctx.fillStyle = g;
    ctx.fillRect(0, 0, w, h);

    // depth-based darkening (cheap shadow feel): darken far areas
    ctx.globalCompositeOperation = 'multiply';
    ctx.globalAlpha = uniforms.ambient * 0.5;
    ctx.drawImage(depthCanvas, 0, 0, w, h);
    ctx.globalAlpha = 1;
    ctx.globalCompositeOperation = 'source-over';
  }
}
