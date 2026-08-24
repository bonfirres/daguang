# 空手挪灯 · 实时深度打光 (Hand-light, real-time depth lighting)

复刻抖音视频里的效果：**普通摄像头进画面 → DINOv2 估深度 → 手指当灯（跟手）→ WebGPU 把光和影子打回实景**。

技术上对应原视频的每一环：

| 原视频 | 本复刻实现 |
| --- | --- |
| 普通 USB / 手机摄像头 | `getUserMedia` 网页摄像头 |
| Meta FAIR 的 DINOv2 估深度 | `depth-anything-small`（**DINOv2 主干**）经 `@huggingface/transformers` 在浏览器 WebGPU 推理 |
| Apple Vision 跟手 | MediaPipe `HandLandmarker`，取食指尖(lm8)为光源位置，捏合调亮度 |
| 448×336 一帧里深度+手势 < 25ms | 深度推理在 rAF **之外**的异步循环跑；rAF 内只做「采样深度纹理 + 手部追踪 + GPU 打光」，保持低延迟 |
| react-native-webgpu 零拷贝打光 | **TypeGPU** (`software-mansion/TypeGPU`) 取 GPU 设备，原生 WebGPU 全屏管线直接采样摄像头纹理（无 CPU 回读） |

## 运行方式

摄像头需要**安全上下文**（https 或 localhost），且浏览器需支持 WebGPU（Chrome / Edge 113+）。

```bash
# 在项目目录起一个本地静态服务器
cd hand-light-depth
python3 -m http.server 8000
# 浏览器打开
#   http://localhost:8000
```

1. 点「▶ 启动摄像头」，授权摄像头。默认走**亮度近似深度**，**立刻就有效果**（不卡、不下载）。
2. 把手伸进画面，**食指尖就是灯**；张手/捏合调节亮度。没有手时用**鼠标**移动灯。
3. 想用更准的真实 DINOv2：先确认 1–2 步流畅，再勾选「启用真实 DINOv2（本地模型）」，状态栏会显示加载进度，成功显示「✅ DINOv2 真实深度已启用」。
4. 右侧滑块实时调：强度 / 环境光 / 深度→法线强度 / 阴影柔和度 / 光源高度 / 镜像 / 灯光颜色。

> 性能提示：transformers.js 多线程需要 COOP/COEP 响应头才最快。没有也能跑（单线程，稍慢）。
> 如需最快，可在服务器加 `Cross-Origin-Opener-Policy: same-origin` 与 `Cross-Origin-Embedder-Policy: require-corp`。

## 文件

- `index.html` — UI + import map（typegpu / transformers / mediapipe 走 esm.sh CDN）
- `main.js` — 编排：摄像头、双循环（rAF 打光 + 异步深度推理）、鼠标兜底
- `depth.js` — DINOv2 深度估计（`onnx-community/depth-anything-small-hf`）
- `hands.js` — MediaPipe 手部追踪（手指当灯）
- `gpu.js` — TypeGPU/WebGPU 渲染器（WGSL 深度法线 + 漫反射/高光 + 屏幕空间阴影）+ 无 WebGPU 时的 2D 兜底

## 模型下载卡住 / 本地加载真实 DINOv2

`depth.js` 默认**零下载**即可运行：用「亮度+径向」近似深度，光和影的效果立刻可见（不如真 DINOv2 准，但机理一样）。真实模型改为**手动勾选**加载（页面上「启用真实 DINOv2（本地模型）」），这样默认启动绝不会卡；勾选后才在后台加载，成功自动切换。

若想用**真实 DINOv2** 且你的网络到 huggingface.co 不通，可把模型文件放到本地、零外网加载：

> ⚠️ **必须是 ONNX 模型，不是 PyTorch 模型。**
> transformers.js 在浏览器里只跑 ONNX。你在 HF 上看到的那种只含 `model.safetensors` / `pytorch_model.bin`、**没有 `onnx/` 文件夹**的仓库（例如 `LiheYoung/depth-anything-small-hf`）是 PyTorch 版，**下回来也用不了**。
> 正确来源（结构符合 `depth-estimation` 管线：`config.json` + `preprocessor_config.json` + `onnx/model.onnx`）：
> - 主源：`https://huggingface.co/onnx-community/depth-anything-small-hf`
> - 备源：`https://huggingface.co/Xenova/depth-anything-small-hf`
> - 镜像：`https://hf-mirror.com/onnx-community/depth-anything-small-hf`
> 代码已按「主源 → 备源」自动尝试，每个源先 WebGPU 后 WASM 回退。

1. 在项目里建目录 `hand-light-depth/models/depth-anything-small-hf/`，放入以下文件（从 **`onnx-community/depth-anything-small-hf`** 仓库下载）：
   - `config.json`
   - `preprocessor_config.json`
   - `onnx/model.onnx`（fp32，约 99MB）**或** `onnx/model_quantized.onnx`（q8 量化，约 27MB，更轻、更不易卡 GPU，**推荐**）
2. 打开 `depth.js`，把 `const USE_LOCAL = false;` 改成 `true`。若用**量化版**，把 `DEVICE_FALLBACKS` 里第一个的 `dtype: 'fp32'` 改成 `'q8'`。
3. 重启 `python -m http.server` 并刷新页面。先点「▶ 启动摄像头」（默认近似深度，立即出效果）；再勾选「启用真实 DINOv2（本地模型）」，状态栏会显示模型加载进度，成功显示「✅ DINOv2 真实深度已启用」。

> 目录结构应为：`hand-light-depth/models/depth-anything-small-hf/{config.json,preprocessor_config.json,onnx/model.onnx}`（或 `onnx/model_quantized.onnx`），
> 这样 `http://localhost:8000/models/depth-anything-small-hf/config.json` 能直接访问到。

## 与原版的差异 / 已知可调点

- **平台**：原版是 React Native + VisionCamera + Apple Vision + react-native-webgpu 的原生移动栈；本复刻是**浏览器版**（WebGPU + Web），原理一致、可直接跑、可改。
- **深度方向**：Depth-Anything 输出里“近/远”的朝向由 `depthScale` 滑块的符号控制。若发现阴影/法线反了，把 `depthScale` 调到负值即可。
- **阴影**：用屏幕空间 ray-march 近似（沿光线查深度遮挡），是实时近似，不是光线追踪级阴影；用 `shadowSoft` 调柔和度。
- **模型源自动重试**：`depth.js` 里的 `REMOTE_MODEL_IDS` 是候选源列表（默认 `onnx-community/depth-anything-small-hf` → `Xenova/depth-anything-small-hf`），按顺序尝试；且每个源都会先试 **WebGPU**、失败再回退 **WASM(CPU)**。任一个成功就自动切换真实深度。如仍全失败则继续用近似深度，状态栏会列出原因。
- **TypeGPU 用法**：用 `tgpu.init()` 作为 GPU 运行时/设备基础，管线与 WGSL 走原生 WebGPU（TypeGPU 官方支持“随时 eject 到原生 WebGPU”，且无需打包插件即可用）。若环境没有打包步骤，类型化 uniform buffer 会自动回退到原生 `GPUBuffer`，效果一致。
