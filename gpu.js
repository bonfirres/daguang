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
  lightRadius: f32,    // 36
  lightColor : vec3f,  // 48 (align 16)
  videoAspect: f32,    // 60  (native camera aspect, for cover-fit)
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
  let d = in.uv;                                   // display uv (0..1)

  // --- cover-fit: map display uv -> source uv so a landscape webcam fills any
  // box (portrait/square included) without stretching. Rc = display/video.
  let Rc = U.aspect / U.videoAspect;
  var suv = d;
  if (Rc > 1.0) { suv = vec2f(d.x, (d.y - 0.5) / Rc + 0.5); }
  else          { suv = vec2f((d.x - 0.5) * Rc + 0.5, d.y); }
  let sx = suv.x * (1.0 - 2.0 * m) + m;            // mirrored source x
  let sy = suv.y;

  let color = textureSampleLevel(camTex, samp, vec2f(sx, sy), 0.0).rgb;

  let dim = textureDimensions(camTex);
  let texel = vec2f(1.0 / f32(dim.x), 1.0 / f32(dim.y));
  let gx = sampleDepth(vec2f(sx + texel.x, sy)) - sampleDepth(vec2f(sx - texel.x, sy));
  let gy = sampleDepth(vec2f(sx, sy + texel.y)) - sampleDepth(vec2f(sx, sy - texel.y));
  let normal = normalize(vec3f(-gx * U.depthScale, -gy * U.depthScale, 1.0));

  // light position in source uv (cover + mirror of the display light pos)
  let dlight = U.lightPos;                         // display uv
  var luv = dlight;
  if (Rc > 1.0) { luv = vec2f(dlight.x, (dlight.y - 0.5) / Rc + 0.5); }
  else          { luv = vec2f((dlight.x - 0.5) * Rc + 0.5, dlight.y); }
  let lx = luv.x * (1.0 - 2.0 * m) + m;
  let ly = luv.y;

  // lighting geometry in DISPLAY space so the orb/spill stay circular on screen
  let frag  = vec2f(d.x * U.aspect, d.y);
  let light = vec2f(dlight.x * U.aspect, dlight.y);
  var toLight = vec3f(light - frag, U.lightHeight);
  let dist = length(toLight);
  let L = toLight / max(dist, 1e-4);

  // screen-space shadow march from fragment toward the light (source space)
  var occ = 0.0;
  let steps = 18;
  let baseD = sampleDepth(vec2f(sx, sy));
  for (var i = 1; i <= steps; i = i + 1) {
    let t = f32(i) / f32(steps);
    let suv_m = mix(vec2f(sx, sy), vec2f(lx, ly), t);
    let sd = sampleDepth(suv_m);
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

  // --- the lamp: a hot glowing orb + a soft warm spill over the scene ---
  // (matches the demo: bright white core, warm body, big soft halo that
  //  bathes the room in the light's color). Geometry is in DISPLAY space so
  //  the orb stays a circle at any window aspect ratio.
  let dd = vec2f((d.x - dlight.x) * U.aspect, d.y - dlight.y);
  let rr = length(dd);
  let Rb = U.lightRadius;
  if (rr < Rb) {
    let z = sqrt(max(Rb * Rb - rr * rr, 0.0));
    let sn = vec3f(dd / Rb, z / Rb);
    let sdiff = clamp(sn.z, 0.0, 1.0);
    // body: emissive warm sphere
    var sphere = U.lightColor * (0.55 + 0.75 * sdiff);
    // hot white blown-out core
    let hot = pow(sdiff, 32.0);
    sphere = sphere + vec3f(1.0) * hot * 1.3;
    outc = sphere;
  } else {
    // tight warm bloom right around the orb
    let bloom = exp(-(rr - Rb) * 6.0) * U.intensity;
    // wide soft spill that lights the surrounding scene
    let spill = exp(-rr * 3.0) * U.intensity * 0.4;
    outc = outc + U.lightColor * (bloom * 1.0 + spill);
  }

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
    u[9] = uniforms.lightRadius || 0.06;
    u[12] = uniforms.color[0]; u[13] = uniforms.color[1]; u[14] = uniforms.color[2];
    u[15] = uniforms.videoAspect || (16 / 9);   // for cover-fit
    // indices 9,10,11 stay zero (padding)

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
    const cw = this.canvas.width, ch = this.canvas.height;   // display box (layout owns size)
    const vw = video.videoWidth || 640, vh = video.videoHeight || 480;
    // cover-fit the (landscape) camera into the possibly-portrait canvas
    const scale = Math.max(cw / vw, ch / vh);
    const dw = vw * scale, dh = vh * scale;
    const dx = (cw - dw) / 2, dy = (ch - dh) / 2;

    ctx.save();
    if (uniforms.mirror) { ctx.translate(cw, 0); ctx.scale(-1, 1); }
    ctx.drawImage(video, dx, dy, dw, dh);
    ctx.restore();

    // glowing sphere at the hand position (display-space coords)
    const lx = uniforms.mirror ? (1 - uniforms.x) * cw : uniforms.x * cw;
    const ly = uniforms.y * ch;
    const [r, gg, b] = uniforms.color;
    const R = (uniforms.lightRadius || 0.07) * ch;   // ball radius (px)
    const cr = (r * 255) | 0, cg = (gg * 255) | 0, cb = (b * 255) | 0;

    // wide soft warm spill (lights the surrounding scene like the demo)
    const spill = ctx.createRadialGradient(lx, ly, R, lx, ly, R * 8);
    const sa = Math.min(uniforms.intensity * 0.22, 0.55);
    spill.addColorStop(0, `rgba(${cr},${cg},${cb},${sa})`);
    spill.addColorStop(1, 'rgba(0,0,0,0)');
    ctx.globalCompositeOperation = 'lighter';
    ctx.fillStyle = spill;
    ctx.fillRect(0, 0, cw, ch);

    // tight warm bloom around the orb
    const bloom = ctx.createRadialGradient(lx, ly, R * 0.5, lx, ly, R * 3);
    const ba = Math.min(uniforms.intensity * 0.8, 1.0);
    bloom.addColorStop(0, `rgba(${cr},${cg},${cb},${ba})`);
    bloom.addColorStop(0.4, `rgba(${cr},${cg},${cb},${ba * 0.4})`);
    bloom.addColorStop(1, 'rgba(0,0,0,0)');
    ctx.fillStyle = bloom;
    ctx.fillRect(0, 0, cw, ch);

    // orb body: bright white core, warm shell
    const body = ctx.createRadialGradient(lx - R * 0.3, ly - R * 0.3, R * 0.05, lx, ly, R);
    body.addColorStop(0, 'rgba(255,255,255,1)');
    body.addColorStop(0.22, 'rgba(255,250,240,1)');
    body.addColorStop(0.5, `rgba(${cr},${cg},${cb},1)`);
    body.addColorStop(1, `rgba(${cr},${cg},${cb},0.95)`);
    ctx.fillStyle = body;
    ctx.beginPath();
    ctx.arc(lx, ly, R, 0, Math.PI * 2);
    ctx.fill();
    ctx.globalCompositeOperation = 'source-over';

    // depth-based darkening (cheap shadow feel): darken far areas
    ctx.globalCompositeOperation = 'multiply';
    ctx.globalAlpha = uniforms.ambient * 0.5;
    ctx.drawImage(depthCanvas, 0, 0, cw, ch);
    ctx.globalAlpha = 1;
    ctx.globalCompositeOperation = 'source-over';
  }
}
