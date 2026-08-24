// hands.js — real-time hand tracking with MediaPipe Tasks Vision.
//
// Mirrors "Apple Vision 跟手": we track one hand and use the index-fingertip
// landmark (id 8) as the position of the virtual light. A pinch (thumb tip id 4
// near index tip id 8) raises light intensity. If no hand is present the caller
// falls back to the mouse, so the effect is always interactive.

import {
  HandLandmarker,
  FilesetResolver,
} from '@mediapipe/tasks-vision';

const WASM_PATH = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision/wasm';
const MODEL_URL =
  'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task';

let landmarker = null;

export async function loadHandModel(onStatus = () => {}) {
  if (landmarker) return landmarker;
  onStatus('Loading hand model…');
  const fileset = await FilesetResolver.forVisionTasks(WASM_PATH);
  landmarker = await HandLandmarker.createFromOptions(fileset, {
    baseOptions: { modelAssetPath: MODEL_URL, delegate: 'GPU' },
    runningMode: 'VIDEO',
    numHands: 1,
  });
  onStatus('Hand model ready');
  return landmarker;
}

// Returns { present, x, y, intensity } where x,y are normalized image coords
// (0..1, origin top-left) and intensity is 0..1. `now` is a timestamp in ms.
export function detectHands(video, now) {
  if (!landmarker) return { present: false, x: 0.5, y: 0.5, intensity: 0.6 };
  const res = landmarker.detectForVideo(video, now);
  if (!res.landmarks || res.landmarks.length === 0) {
    return { present: false, x: 0.5, y: 0.5, intensity: 0.6 };
  }
  const lm = res.landmarks[0]; // 21 landmarks
  const tip = lm[8]; // index fingertip
  const thumb = lm[4]; // thumb tip
  // Pinch distance in normalized space -> intensity.
  const pinch = Math.hypot(tip.x - thumb.x, tip.y - thumb.y);
  const intensity = Math.min(1, 0.35 + (1 - Math.min(1, pinch / 0.18)) * 0.9);
  return { present: true, x: tip.x, y: tip.y, intensity };
}
