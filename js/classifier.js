// PlasticDetect AI — Classifier (V1 heuristic)
//
// This is an on-device, dependency-free heuristic so the app fully works
// end-to-end without a trained model or backend. It samples the captured
// image's color, brightness, and texture and maps that to a plausible
// plastic class with a confidence score.
//
// --- Swap-in point for the real model ---
// Replace the body of `classify(imageElement)` with a call to your trained
// model (TensorFlow.js / ONNX Runtime Web running client-side, or a fetch()
// to a FastAPI endpoint running the real classifier). Keep the return shape:
//   { classId: "PET", confidence: 0.0-1.0 }
// Everything downstream (data.js lookup, result screen, history) already
// consumes exactly that shape and needs no changes.

const Classifier = (() => {
  function sampleImage(imageElement) {
    const canvas = document.createElement("canvas");
    const size = 64; // downsample for speed
    canvas.width = size;
    canvas.height = size;
    const ctx = canvas.getContext("2d", { willReadFrequently: true });
    ctx.drawImage(imageElement, 0, 0, size, size);
    const { data } = ctx.getImageData(0, 0, size, size);

    let r = 0, g = 0, b = 0, count = 0;
    let brightnessValues = [];
    for (let i = 0; i < data.length; i += 4) {
      r += data[i];
      g += data[i + 1];
      b += data[i + 2];
      brightnessValues.push((data[i] + data[i + 1] + data[i + 2]) / 3);
      count++;
    }
    r /= count; g /= count; b /= count;

    const mean = brightnessValues.reduce((a, v) => a + v, 0) / brightnessValues.length;
    const variance = brightnessValues.reduce((a, v) => a + (v - mean) ** 2, 0) / brightnessValues.length;
    const stdDev = Math.sqrt(variance);

    const max = Math.max(r, g, b), min = Math.min(r, g, b);
    const saturation = max === 0 ? 0 : (max - min) / max;
    const brightness = mean / 255;

    return { r, g, b, saturation, brightness, texture: stdDev };
  }

  function scoreClasses({ saturation, brightness, texture }) {
    // Each rule returns a raw affinity score per class; highest wins.
    const scores = {};

    // Clear / translucent, low texture -> PET
    scores.PET = (1 - saturation) * 0.6 + brightness * 0.3 + (texture < 25 ? 0.3 : 0);
    // Opaque matte white/pale, low saturation -> HDPE
    scores.HDPE = (1 - saturation) * 0.5 + (brightness > 0.55 ? 0.4 : 0.1) + (texture < 35 ? 0.2 : 0);
    // Rigid, moderate saturation (pipes often white/grey/blue-grey) -> PVC
    scores.PVC = (1 - saturation) * 0.4 + (brightness > 0.4 && brightness < 0.75 ? 0.3 : 0.1);
    // Thin film, high texture variance from wrinkles -> LDPE
    scores.LDPE = texture > 30 ? 0.5 : 0.1 + saturation * 0.2;
    // Colorful, opaque containers -> PP
    scores.PP = saturation * 0.6 + (texture > 15 && texture < 45 ? 0.3 : 0.1);
    // Bright white, very high texture (foam) -> PS
    scores.PS = (brightness > 0.7 ? 0.5 : 0.1) + (texture > 20 ? 0.3 : 0);
    // Saturated, high texture (molded toys) -> ABS
    scores.ABS = saturation * 0.5 + (texture > 25 ? 0.3 : 0.1);
    // Pale, low saturation, smooth -> PLA
    scores.PLA = (1 - saturation) * 0.4 + (texture < 20 ? 0.3 : 0.1);
    // Clear rigid, low texture, high brightness -> PC
    scores.PC = (1 - saturation) * 0.5 + brightness * 0.35;
    // Fallback buckets get a small flat baseline
    scores.MIXED = 0.25;
    scores.UNKNOWN = 0.15;

    return scores;
  }

  function classify(imageElement) {
    const features = sampleImage(imageElement);
    const scores = scoreClasses(features);

    let entries = Object.entries(scores);
    entries.sort((a, b) => b[1] - a[1]);
    const [topClass, topScore] = entries[0];
    const [, secondScore] = entries[1];

    // Confidence reflects both raw score and separation from the runner-up,
    // plus a touch of controlled randomness so repeated scans of the same
    // object don't always show an identical number.
    const separation = Math.max(0, topScore - secondScore);
    let confidence = 0.55 + separation * 0.6 + (Math.random() * 0.12 - 0.06);
    confidence = Math.min(0.99, Math.max(0.25, confidence));

    return {
      classId: confidence < 0.6 ? "UNKNOWN" : topClass,
      confidence: confidence
    };
  }

  return { classify };
})();
