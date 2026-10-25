"""
FaceGuard AI - Interactive Web Dashboard
Monash University MCS24 Team | Deep Learning Project
Runs standalone using Python's built-in HTTP server (zero extra pip installs needed!)
"""

import os
import io
import json
import base64
import time
import webbrowser
from http.server import HTTPServer, BaseHTTPRequestHandler
import torch
from PIL import Image
from torchvision import transforms

from detectors.xception_detector import XceptionDetector

# -------------------------------------------------------------
# 1. Model Initialization
# -------------------------------------------------------------
print("=" * 60)
print("  Initializing FaceGuard AI Deepfake Detector...")
print("=" * 60)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[FaceGuard] Hardware acceleration: {device}")

model = XceptionDetector()

checkpoint_path = os.path.abspath("../xception.pth")
if not os.path.exists(checkpoint_path):
    # Fallback to local directory if run from root
    checkpoint_path = os.path.abspath("xception.pth")

print(f"[FaceGuard] Loading checkpoint: {checkpoint_path}")
checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
    checkpoint = checkpoint["state_dict"]
elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
    checkpoint = checkpoint["model_state_dict"]

model.load_state_dict(checkpoint)
model.to(device)
model.eval()
print("[FaceGuard] Deepfake detection model ready!")

# Transform pipeline matching training
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])

# -------------------------------------------------------------
# 2. Prediction Function
# -------------------------------------------------------------
def predict_image(image_bytes):
    start_time = time.time()
    
    # 1. Ephemeral in-memory load (Privacy-by-Design: no disk write)
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    orig_w, orig_h = image.size
    
    # 2. Transform into tensor
    tensor = transform(image).unsqueeze(0).to(device)
    
    # 3. Model forward pass
    with torch.inference_mode():
        result = model({"image": tensor}, inference=True)
        logit = result["cls"].reshape(-1)[0]
        fake_prob = torch.sigmoid(logit).item()
    
    latency_ms = round((time.time() - start_time) * 1000, 1)
    real_prob = 1.0 - fake_prob
    is_fake = fake_prob >= 0.5
    confidence = fake_prob if is_fake else real_prob
    
    return {
        "is_fake": is_fake,
        "prediction": "AI-GENERATED / FAKE" if is_fake else "AUTHENTIC REAL HUMAN",
        "fake_probability": round(fake_prob * 100, 2),
        "real_probability": round(real_prob * 100, 2),
        "confidence": round(confidence * 100, 2),
        "raw_logit": round(logit.item(), 4),
        "latency_ms": latency_ms,
        "resolution": f"{orig_w}x{orig_h}",
        "device": str(device)
    }

# -------------------------------------------------------------
# 3. Modern Embedded Web Dashboard (HTML + Tailwind CSS)
# -------------------------------------------------------------
HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>FaceGuard AI | Deepfake Profile Photo Detector</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
  <style>
    body { font-family: 'Inter', sans-serif; }
    .mono { font-family: 'JetBrains Mono', monospace; }
    @keyframes pulse-ring {
      0% { transform: scale(0.95); opacity: 0.8; }
      50% { transform: scale(1.05); opacity: 0.3; }
      100% { transform: scale(0.95); opacity: 0.8; }
    }
    .scanner-line {
      height: 2px;
      background: linear-gradient(90deg, transparent, #38bdf8, transparent);
      position: absolute;
      width: 100%;
      top: 0;
      animation: scan 2s cubic-bezier(0.4, 0, 0.2, 1) infinite;
    }
    @keyframes scan {
      0% { top: 0%; opacity: 0; }
      20% { opacity: 1; }
      80% { opacity: 1; }
      100% { top: 100%; opacity: 0; }
    }
  </style>
</head>
<body class="bg-slate-950 text-slate-100 min-h-screen flex flex-col antialiased selection:bg-cyan-500 selection:text-white">

  <!-- Header -->
  <header class="border-b border-slate-800 bg-slate-900/60 backdrop-blur-md sticky top-0 z-50">
    <div class="max-w-6xl mx-auto px-6 py-4 flex items-center justify-between">
      <div class="flex items-center space-x-3">
        <div class="h-10 w-10 rounded-xl bg-gradient-to-tr from-cyan-600 to-blue-500 flex items-center justify-center shadow-lg shadow-cyan-500/20">
          <svg class="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z"/>
          </svg>
        </div>
        <div>
          <div class="flex items-center space-x-2">
            <span class="text-xl font-bold tracking-tight bg-gradient-to-r from-white via-slate-100 to-cyan-300 bg-clip-text text-transparent">FaceGuard AI</span>
            <span class="text-xs px-2 py-0.5 rounded-full bg-cyan-500/10 text-cyan-400 font-semibold border border-cyan-500/20">v2.0</span>
          </div>
          <p class="text-xs text-slate-400">Privacy-Preserving Deepfake Profile Detector</p>
        </div>
      </div>
      <div class="flex items-center space-x-3">
        <span class="hidden sm:inline-flex items-center px-3 py-1 rounded-full text-xs font-medium bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
          <span class="w-1.5 h-1.5 rounded-full bg-emerald-400 mr-2 animate-pulse"></span>
          Model Loaded: Xception (ImageNet + AI-Face)
        </span>
      </div>
    </div>
  </header>

  <!-- Main Content -->
  <main class="flex-1 max-w-6xl w-full mx-auto px-6 py-8">
    
    <!-- Hero / Info Banner -->
    <div class="mb-8 p-5 rounded-2xl bg-gradient-to-r from-slate-900 via-slate-900 to-cyan-950/40 border border-slate-800">
      <div class="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
        <div>
          <h2 class="text-lg font-semibold text-white">AI Face Authenticity Verification</h2>
          <p class="text-sm text-slate-400 mt-1 max-w-2xl">
            Detects GAN-generated (StyleGAN, ProGAN) and Diffusion-synthesized (Stable Diffusion, Midjourney) human profile images with deep convolutional feature forensics.
          </p>
        </div>
        <div class="flex flex-wrap gap-2 text-xs">
          <span class="px-2.5 py-1 rounded-lg bg-slate-800/80 text-slate-300 border border-slate-700">🔒 Ephemeral In-Memory</span>
          <span class="px-2.5 py-1 rounded-lg bg-slate-800/80 text-slate-300 border border-slate-700">🛡️ Zero Disk Retention</span>
          <span class="px-2.5 py-1 rounded-lg bg-slate-800/80 text-slate-300 border border-slate-700">⚡ 256x256 Forensics</span>
        </div>
      </div>
    </div>

    <!-- 2 Column Layout: Upload Box & Results Panel -->
    <div class="grid grid-cols-1 lg:grid-cols-2 gap-8 items-start">
      
      <!-- Left: Upload & Input -->
      <div class="space-y-6">
        <div class="bg-slate-900/80 border border-slate-800 rounded-3xl p-6 shadow-xl">
          <h3 class="text-sm font-semibold text-slate-300 uppercase tracking-wider mb-4 flex items-center">
            <svg class="w-4 h-4 mr-2 text-cyan-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z"/>
            </svg>
            Upload Profile Picture
          </h3>

          <!-- Dropzone -->
          <div id="dropzone" class="relative group border-2 border-dashed border-slate-700 hover:border-cyan-500 transition-all duration-300 rounded-2xl p-8 text-center cursor-pointer bg-slate-950/40 hover:bg-cyan-950/10">
            <input type="file" id="fileInput" accept="image/jpeg,image/png,image/webp,image/jpg" class="hidden">
            
            <div id="uploadPrompt" class="space-y-3">
              <div class="w-16 h-16 mx-auto rounded-2xl bg-cyan-500/10 border border-cyan-500/20 flex items-center justify-center text-cyan-400 group-hover:scale-105 transition-transform duration-300">
                <svg class="w-8 h-8" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"/>
                </svg>
              </div>
              <div>
                <p class="text-sm font-medium text-slate-200"><span class="text-cyan-400 underline decoration-cyan-500/40 underline-offset-4">Click to browse</span> or drag and drop</p>
                <p class="text-xs text-slate-500 mt-1">Supports JPG, PNG, WEBP</p>
              </div>
            </div>

            <!-- Image Preview Box -->
            <div id="previewContainer" class="hidden relative rounded-xl overflow-hidden max-h-72 flex items-center justify-center bg-black/40">
              <img id="imagePreview" src="" alt="Preview" class="max-h-72 w-auto object-contain rounded-xl">
              <div id="scannerLine" class="scanner-line hidden"></div>
            </div>
          </div>

          <!-- Face Cropping Tip -->
          <div class="mt-4 p-3 rounded-xl bg-slate-800/40 border border-slate-700/60 text-xs text-slate-400 flex items-start space-x-2.5">
            <svg class="w-4 h-4 text-cyan-400 shrink-0 mt-0.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"/>
            </svg>
            <span><strong>Tip for Best Accuracy:</strong> Like all Deepfake models, FaceGuard inspects micro-facial artifacts. Upload photos where the face is centered and fills the frame.</span>
          </div>

          <!-- Buttons -->
          <div class="mt-6 flex space-x-3">
            <button id="analyzeBtn" disabled class="flex-1 py-3 px-4 rounded-xl font-semibold text-sm bg-gradient-to-r from-cyan-500 to-blue-600 hover:from-cyan-400 hover:to-blue-500 text-white shadow-lg shadow-cyan-500/20 disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-200 flex items-center justify-center space-x-2">
              <svg id="btnSpinner" class="hidden w-4 h-4 animate-spin" fill="none" viewBox="0 0 24 24">
                <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
                <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
              </svg>
              <span id="btnText">Analyze Authenticity</span>
            </button>
            <button id="resetBtn" class="py-3 px-4 rounded-xl font-medium text-sm bg-slate-800 hover:bg-slate-700 text-slate-300 transition-colors">
              Reset
            </button>
          </div>
        </div>
      </div>

      <!-- Right: Forensic Analysis Results -->
      <div>
        <div id="resultsCard" class="bg-slate-900/80 border border-slate-800 rounded-3xl p-6 shadow-xl min-h-[460px] flex flex-col justify-between">
          <div>
            <h3 class="text-sm font-semibold text-slate-300 uppercase tracking-wider mb-4 flex items-center">
              <svg class="w-4 h-4 mr-2 text-cyan-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"/>
              </svg>
              Forensic Results
            </h3>

            <!-- Placeholder State -->
            <div id="emptyState" class="py-20 text-center space-y-3">
              <div class="w-16 h-16 mx-auto rounded-2xl bg-slate-800/60 border border-slate-700 flex items-center justify-center text-slate-500">
                <svg class="w-8 h-8" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2"/>
                </svg>
              </div>
              <p class="text-sm text-slate-400 font-medium">No Image Analyzed Yet</p>
              <p class="text-xs text-slate-500 max-w-xs mx-auto">Upload a human face photo on the left and click "Analyze Authenticity" to start deepfake detection.</p>
            </div>

            <!-- Active Analysis Result (Hidden Initially) -->
            <div id="outputContainer" class="hidden space-y-6">
              
              <!-- Big Verdict Banner -->
              <div id="verdictBanner" class="p-6 rounded-2xl border text-center transition-all duration-300">
                <span id="verdictSubtitle" class="text-xs font-bold tracking-widest uppercase block mb-1">Authenticity Verdict</span>
                <div id="verdictTitle" class="text-2xl font-black tracking-tight">REAL PROFILE</div>
                <div id="verdictConfidence" class="text-sm font-semibold mt-1">99.50% Confidence</div>
              </div>

              <!-- Probability Bars -->
              <div class="space-y-4 bg-slate-950/40 p-4 rounded-2xl border border-slate-800/80">
                <div>
                  <div class="flex justify-between text-xs font-semibold mb-1.5">
                    <span class="text-emerald-400 flex items-center">
                      <span class="w-2 h-2 rounded-full bg-emerald-400 mr-1.5"></span>
                      Real Human Probability
                    </span>
                    <span id="realPct" class="mono text-slate-200">0.00%</span>
                  </div>
                  <div class="h-3 rounded-full bg-slate-800 overflow-hidden">
                    <div id="realBar" class="h-full bg-emerald-500 rounded-full transition-all duration-700 ease-out" style="width: 0%"></div>
                  </div>
                </div>

                <div>
                  <div class="flex justify-between text-xs font-semibold mb-1.5">
                    <span class="text-rose-400 flex items-center">
                      <span class="w-2 h-2 rounded-full bg-rose-400 mr-1.5"></span>
                      AI-Generated Probability
                    </span>
                    <span id="fakePct" class="mono text-slate-200">0.00%</span>
                  </div>
                  <div class="h-3 rounded-full bg-slate-800 overflow-hidden">
                    <div id="fakeBar" class="h-full bg-rose-500 rounded-full transition-all duration-700 ease-out" style="width: 0%"></div>
                  </div>
                </div>
              </div>

              <!-- Neural Network Stats -->
              <div class="grid grid-cols-2 gap-3 text-xs">
                <div class="p-3 rounded-xl bg-slate-950/60 border border-slate-800/80">
                  <span class="text-slate-400 block mb-0.5">Raw Classifier Logit</span>
                  <span id="statLogit" class="mono font-semibold text-slate-200 text-sm">--</span>
                </div>
                <div class="p-3 rounded-xl bg-slate-950/60 border border-slate-800/80">
                  <span class="text-slate-400 block mb-0.5">Inference Latency</span>
                  <span id="statLatency" class="mono font-semibold text-cyan-400 text-sm">--</span>
                </div>
                <div class="p-3 rounded-xl bg-slate-950/60 border border-slate-800/80">
                  <span class="text-slate-400 block mb-0.5">Original Resolution</span>
                  <span id="statRes" class="mono font-semibold text-slate-200 text-sm">--</span>
                </div>
                <div class="p-3 rounded-xl bg-slate-950/60 border border-slate-800/80">
                  <span class="text-slate-400 block mb-0.5">Execution Device</span>
                  <span id="statDevice" class="mono font-semibold text-slate-200 text-sm">--</span>
                </div>
              </div>

            </div>
          </div>

          <!-- Footer Badge -->
          <div class="pt-4 mt-6 border-t border-slate-800/80 flex items-center justify-between text-xs text-slate-500">
            <span>Powered by PyTorch & Xception</span>
            <span>Monash MCS24 Team</span>
          </div>

        </div>
      </div>

    </div>
  </main>

  <script>
    const dropzone = document.getElementById('dropzone');
    const fileInput = document.getElementById('fileInput');
    const uploadPrompt = document.getElementById('uploadPrompt');
    const previewContainer = document.getElementById('previewContainer');
    const imagePreview = document.getElementById('imagePreview');
    const analyzeBtn = document.getElementById('analyzeBtn');
    const resetBtn = document.getElementById('resetBtn');
    const btnSpinner = document.getElementById('btnSpinner');
    const btnText = document.getElementById('btnText');
    const scannerLine = document.getElementById('scannerLine');

    const emptyState = document.getElementById('emptyState');
    const outputContainer = document.getElementById('outputContainer');
    const verdictBanner = document.getElementById('verdictBanner');
    const verdictTitle = document.getElementById('verdictTitle');
    const verdictSubtitle = document.getElementById('verdictSubtitle');
    const verdictConfidence = document.getElementById('verdictConfidence');
    const realPct = document.getElementById('realPct');
    const realBar = document.getElementById('realBar');
    const fakePct = document.getElementById('fakePct');
    const fakeBar = document.getElementById('fakeBar');
    const statLogit = document.getElementById('statLogit');
    const statLatency = document.getElementById('statLatency');
    const statRes = document.getElementById('statRes');
    const statDevice = document.getElementById('statDevice');

    let currentBase64 = null;

    // Handle Dropzone Click
    dropzone.addEventListener('click', () => fileInput.click());

    // File Drag & Drop
    ['dragenter', 'dragover'].forEach(name => {
      dropzone.addEventListener(name, (e) => {
        e.preventDefault();
        dropzone.classList.add('border-cyan-400', 'bg-cyan-950/20');
      });
    });
    ['dragleave', 'drop'].forEach(name => {
      dropzone.addEventListener(name, (e) => {
        e.preventDefault();
        dropzone.classList.remove('border-cyan-400', 'bg-cyan-950/20');
      });
    });
    dropzone.addEventListener('drop', (e) => {
      const files = e.dataTransfer.files;
      if (files.length > 0) handleFile(files[0]);
    });

    fileInput.addEventListener('change', (e) => {
      if (e.target.files.length > 0) handleFile(e.target.files[0]);
    });

    function handleFile(file) {
      if (!file.type.startsWith('image/')) {
        alert('Please upload a valid image file (JPG, PNG, WEBP)');
        return;
      }
      const reader = new FileReader();
      reader.onload = (e) => {
        currentBase64 = e.target.result;
        imagePreview.src = currentBase64;
        uploadPrompt.classList.add('hidden');
        previewContainer.classList.remove('hidden');
        analyzeBtn.disabled = false;
      };
      reader.readAsDataURL(file);
    }

    resetBtn.addEventListener('click', () => {
      currentBase64 = null;
      fileInput.value = '';
      imagePreview.src = '';
      previewContainer.classList.add('hidden');
      uploadPrompt.classList.remove('hidden');
      analyzeBtn.disabled = true;
      outputContainer.classList.add('hidden');
      emptyState.classList.remove('hidden');
      scannerLine.classList.add('hidden');
    });

    analyzeBtn.addEventListener('click', async () => {
      if (!currentBase64) return;

      // Loading state
      analyzeBtn.disabled = true;
      btnSpinner.classList.remove('hidden');
      btnText.innerText = 'Scanning Face...';
      scannerLine.classList.remove('hidden');

      try {
        const response = await fetch('/api/predict', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ image: currentBase64 })
        });

        if (!response.ok) {
          throw new Error('Server returned ' + response.status);
        }

        const data = await response.json();
        renderResult(data);

      } catch (err) {
        alert('Prediction error: ' + err.message);
      } finally {
        analyzeBtn.disabled = false;
        btnSpinner.classList.add('hidden');
        btnText.innerText = 'Analyze Authenticity';
        scannerLine.classList.add('hidden');
      }
    });

    function renderResult(data) {
      emptyState.classList.add('hidden');
      outputContainer.classList.remove('hidden');

      if (data.is_fake) {
        verdictBanner.className = 'p-6 rounded-2xl border text-center transition-all duration-300 bg-rose-500/10 border-rose-500/30 text-rose-300';
        verdictTitle.innerText = 'AI-GENERATED / FAKE';
        verdictTitle.className = 'text-2xl font-black tracking-tight text-rose-400';
        verdictSubtitle.innerText = 'Authenticity Alert: Synthetic Face Detected';
        verdictConfidence.innerText = data.fake_probability + '% Synthetic Confidence';
      } else {
        verdictBanner.className = 'p-6 rounded-2xl border text-center transition-all duration-300 bg-emerald-500/10 border-emerald-500/30 text-emerald-300';
        verdictTitle.innerText = 'AUTHENTIC REAL HUMAN';
        verdictTitle.className = 'text-2xl font-black tracking-tight text-emerald-400';
        verdictSubtitle.innerText = 'Verified Authentic Profile Picture';
        verdictConfidence.innerText = data.real_probability + '% Real Confidence';
      }

      realPct.innerText = data.real_probability + '%';
      realBar.style.width = data.real_probability + '%';

      fakePct.innerText = data.fake_probability + '%';
      fakeBar.style.width = data.fake_probability + '%';

      statLogit.innerText = data.raw_logit;
      statLatency.innerText = data.latency_ms + ' ms';
      statRes.innerText = data.resolution;
      statDevice.innerText = data.device.toUpperCase();
    }
  </script>
</body>
</html>
"""

# -------------------------------------------------------------
# 4. Built-in HTTP Request Handler
# -------------------------------------------------------------
class FaceGuardHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        # Clean logging
        print(f"[FaceGuard Server] {self.command} {self.path} - {args[1]}")

    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML_PAGE.encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path == "/api/predict":
            try:
                content_length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(content_length).decode("utf-8")
                payload = json.loads(body)

                base64_img = payload.get("image", "")
                if "," in base64_img:
                    base64_img = base64_img.split(",")[1]

                img_bytes = base64.b64decode(base64_img)
                result = predict_image(img_bytes)

                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(result).encode("utf-8"))

            except Exception as e:
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()

# -------------------------------------------------------------
# 5. Main Server Runner
# -------------------------------------------------------------
def run(port=8000):
    server_address = ("127.0.0.1", port)
    httpd = HTTPServer(server_address, FaceGuardHandler)
    url = f"http://127.0.0.1:{port}"
    print("\n" + "=" * 60)
    print(f"  🚀 FaceGuard Web Dashboard is LIVE!")
    print(f"  👉 Open in your browser: {url}")
    print("=" * 60)
    print("  Press Ctrl+C in this terminal to stop the server.\n")
    
    try:
        webbrowser.open(url)
    except Exception:
        pass
        
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[FaceGuard] Shutting down web server...")
        httpd.server_close()

if __name__ == "__main__":
    run(port=8000)

