import streamlit as st
from PIL import Image
from model_helper import predict


def to_percent(value):
    # Supports both 0-1 and 0-100 model outputs.
    return value * 100 if value <= 1 else value


def to_progress(value):
    percent = to_percent(value)
    return max(0.0, min(1.0, percent / 100.0))

st.set_page_config(
    page_title="Vehicle Damage Detector",
    page_icon="🚗",
    layout="wide"
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;500;700&family=Space+Grotesk:wght@600;700&display=swap');
    :root {
        --bg1: #fff8e6;
        --bg2: #dff4ff;
        --bg3: #ffe7f2;
        --card: rgba(255, 255, 255, 0.88);
        --text: #10243d;
        --muted: #3d5878;
        --accent: #e85d04;
        --accent2: #0b8f8a;
        --accent3: #5b4bff;
        --good: #1fa866;
    }

    .stApp {
        font-family: 'Outfit', sans-serif;
        background:
            radial-gradient(circle at 12% 15%, #fff3b8 0%, transparent 35%),
            radial-gradient(circle at 86% 20%, #c8e9ff 0%, transparent 38%),
            radial-gradient(circle at 88% 80%, #ffd4ea 0%, transparent 35%),
            linear-gradient(135deg, var(--bg1), var(--bg2) 48%, var(--bg3));
    }

    .block-container {
        padding-top: 1.2rem;
    }

    .hero {
        background: var(--card);
        border: 1px solid #f2dcd2;
        border-radius: 24px;
        padding: 1.25rem 1.5rem;
        box-shadow: 0 16px 36px rgba(35, 64, 110, 0.14);
        margin-bottom: 1rem;
        animation: rise 0.55s ease-out;
    }

    .hero h1 {
        margin: 0;
        color: var(--text);
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 700;
        letter-spacing: 0.2px;
        font-size: clamp(1.7rem, 2.8vw, 2.5rem);
    }

    .hero p {
        margin-top: 0.45rem;
        color: var(--muted);
        font-size: 1rem;
    }

    .chip-row {
        display: flex;
        gap: 0.5rem;
        margin-top: 0.8rem;
        flex-wrap: wrap;
    }

    .chip {
        border-radius: 999px;
        padding: 0.28rem 0.78rem;
        font-weight: 600;
        font-size: 0.82rem;
        color: #17314a;
        background: linear-gradient(90deg, #ffe6c8, #ffe9f2);
        border: 1px solid #f2c9a3;
    }

    @keyframes rise {
        from { opacity: 0; transform: translateY(8px); }
        to { opacity: 1; transform: translateY(0); }
    }

    .result-card {
        background: linear-gradient(145deg, #ffffff, #eef8ff);
        border: 1px solid #d7e4ff;
        border-radius: 16px;
        padding: 1.05rem 1.1rem;
        box-shadow: 0 10px 24px rgba(15, 53, 96, 0.12);
    }

    .metric-title {
        font-size: 0.85rem;
        color: #557096;
        margin-bottom: 0.2rem;
        text-transform: uppercase;
        letter-spacing: 0.7px;
    }

    .metric-value {
        font-family: 'Space Grotesk', sans-serif;
        color: var(--text);
        font-size: 1.3rem;
        font-weight: 700;
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #fff0d8 0%, #e8f4ff 55%, #ffe7f2 100%);
        border-right: 1px solid #f3d9bf;
    }

    section[data-testid="stSidebar"] * {
        color: #163356;
    }

    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, var(--accent), var(--accent2), var(--accent3));
    }

    @media (max-width: 768px) {
        .hero { padding: 1rem; }
        .hero h1 { font-size: 1.5rem; }
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="hero">
        <h1>Vehicle Damage Detection</h1>
        <p>Upload a vehicle photo and get a fast damage category prediction with confidence scores.</p>
        <div class="chip-row">
            <span class="chip">Fast Inference</span>
            <span class="chip">6 Damage Classes</span>
            <span class="chip">Confidence Breakdown</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

with st.sidebar:
    st.subheader("How To Use")
    st.write("1. Upload a clear vehicle image.")
    st.write("2. Front or rear three-quarter angle images perform best.")
    st.write("3. Review the predicted label and per-class confidence scores.")
    st.caption("Supported formats: JPG, JPEG, PNG, WEBP")

uploaded_file = st.file_uploader("Upload Vehicle Image", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file:
    left, right = st.columns([1.2, 1], gap="large")

    try:
        image = Image.open(uploaded_file).convert("RGB")

        with left:
            st.image(image, caption="Uploaded Image", use_column_width=True)

        with right:
            with st.spinner("Analyzing image..."):
                result = predict(image)

            st.markdown('<div class="result-card">', unsafe_allow_html=True)
            st.markdown('<div class="metric-title">Predicted Class</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-value">{result["label"]}</div>', unsafe_allow_html=True)
            st.write("")
            st.markdown('<div class="metric-title">Model Confidence</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-value" style="color: var(--good);">{to_percent(result["confidence"]):.2f}%</div>', unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
            st.write("")
            st.subheader("Class Probabilities")

            ranked = sorted(result["probabilities"].items(), key=lambda item: item[1], reverse=True)
            for cls_name, prob in ranked:
                st.write(f"{cls_name}: {to_percent(prob):.2f}%")
                st.progress(to_progress(prob))

    except Exception as exc:
        st.error("Prediction failed. Please verify the model file, dependencies, and uploaded image format.")
        with st.expander("Technical Details"):
            st.exception(exc)
else:
    st.info("Start by uploading a vehicle image (JPG/JPEG/PNG/WEBP).")
