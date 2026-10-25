"""
FaceGuard AI - Streamlit Dashboard (Alternative)
Run with: streamlit run streamlit_app.py
"""

import os
import io
import time
import torch
from PIL import Image
from torchvision import transforms
import streamlit as st

from detectors.xception_detector import XceptionDetector

st.set_page_config(
    page_title="FaceGuard AI | Deepfake Detector",
    page_icon="🛡️",
    layout="wide"
)

# Custom Styling
st.markdown("""
<style>
    .main { background-color: #0b0f19; }
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #06b6d4, #2563eb);
        color: white;
        border: none;
        padding: 0.75rem;
        font-weight: 600;
        border-radius: 0.75rem;
    }
</style>
""", unsafe_allow_html=True)

st.title("🛡️ FaceGuard AI: Deepfake Profile Detector")
st.caption("Privacy-Preserving Deep Learning Model for Detecting AI-Generated Faces | FIT3161 Project")

@st.cache_resource
def load_detector():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = XceptionDetector()
    checkpoint_path = os.path.abspath("../xception.pth")
    if not os.path.exists(checkpoint_path):
        checkpoint_path = os.path.abspath("xception.pth")
        
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        checkpoint = checkpoint["model_state_dict"]
        
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    return model, device

model, device = load_detector()

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])

col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("📤 Upload Image")
    uploaded_file = st.file_uploader("Upload face photo (JPG, PNG, WEBP)", type=["jpg", "jpeg", "png", "webp"])
    
    st.info("💡 **Tip:** For optimal accuracy, upload tightly cropped face images where the face fills the frame.")
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image Preview", use_container_width=True)
        analyze_btn = st.button("🔍 Analyze Authenticity")

with col2:
    st.subheader("📊 Forensic Results")
    if uploaded_file is not None and 'analyze_btn' in locals() and analyze_btn:
        with st.spinner("Executing neural forensics scan..."):
            start = time.time()
            tensor = transform(image).unsqueeze(0).to(device)
            with torch.inference_mode():
                out = model({"image": tensor}, inference=True)
                logit = out["cls"].reshape(-1)[0]
                fake_prob = torch.sigmoid(logit).item()
                
            latency = (time.time() - start) * 1000
            real_prob = 1.0 - fake_prob
            is_fake = fake_prob >= 0.5
            
        if is_fake:
            st.error(f"⚠️ **AI-GENERATED / FAKE** ({fake_prob*100:.2f}% Confidence)")
        else:
            st.success(f"✅ **AUTHENTIC REAL HUMAN** ({real_prob*100:.2f}% Confidence)")
            
        st.write("---")
        st.write(f"**Real Probability:** {real_prob*100:.2f}%")
        st.progress(real_prob)
        
        st.write(f"**Fake Probability:** {fake_prob*100:.2f}%")
        st.progress(fake_prob)
        
        st.write("---")
        m1, m2 = st.columns(2)
        m1.metric("Raw Logit", f"{logit.item():.4f}")
        m2.metric("Inference Latency", f"{latency:.1f} ms")
    else:
        st.write("Upload an image on the left and click **Analyze Authenticity** to inspect results.")

