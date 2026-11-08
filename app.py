from __future__ import annotations

import base64
from pathlib import Path

import streamlit as st
from PIL import Image

from diagnoai.constants import SYMPTOM_NAMES
from diagnoai.inference import ModelNotReadyError, Predictor


st.set_page_config(
    page_title="DiagnoAI",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');
    :root {
        --bg:#f5f8fc; --ink:#17243a; --muted:#68788e;
        --line:#e2e9f1; --teal:#087f78; --teal-soft:#e7f6f3;
        --panel:#ffffff; --blue:#3459a8;
    }
    html, body, [class*="css"] {font-family:'DM Sans',sans-serif;}
    .stApp {background:var(--bg); color:var(--ink);}
    .stApp label, .stApp p, .stApp caption,
    .stApp [data-testid="stCaptionContainer"],
    .stApp [data-testid="stMetricLabel"],
    .stApp [data-testid="stMetricValue"],
    .stApp [data-testid="stTable"] *,
    .stApp [data-testid="stDataFrame"] * {
        color:#000000 !important;
    }
    [data-testid="stHeader"] {background:rgba(245,248,252,.92);}
    [data-testid="stSidebar"], [data-testid="collapsedControl"] {display:none !important;}
    .block-container {max-width:1180px;padding-top:2rem;padding-bottom:3rem;}
    h1,h2,h3 {font-family:'Manrope',sans-serif !important;color:var(--ink);}
    .hero {
        background:linear-gradient(115deg,#123e50 0%,#176f70 58%,#2d8290 100%);
        border-radius:24px;padding:2rem 2.2rem;color:white;
        box-shadow:0 16px 40px rgba(20,75,91,.14);
    }
    .hero-kicker {font-size:.74rem;letter-spacing:.15em;text-transform:uppercase;
        color:#b9e8e2;font-weight:700;margin-bottom:.65rem;}
    .hero-title {font-family:'Manrope',sans-serif;font-size:clamp(2rem,4vw,3rem);
        line-height:1.1;letter-spacing:-.045em;font-weight:800;color:white;}
    .hero-copy {color:#d8efed;max-width:720px;line-height:1.6;margin-top:.7rem;}
    .section-card {background:var(--panel);border:1px solid var(--line);
        border-radius:18px;padding:1.25rem 1.35rem;margin-bottom:1rem;
        box-shadow:0 5px 18px rgba(24,45,75,.035);}
    .step {font-size:.72rem;font-weight:800;letter-spacing:.13em;
        color:var(--teal);text-transform:uppercase;margin-bottom:.35rem;}
    .section-title {font-family:'Manrope',sans-serif;font-size:1.28rem;
        font-weight:800;color:var(--ink);letter-spacing:-.025em;}
    .section-help {color:var(--muted);font-size:.9rem;line-height:1.55;margin:.35rem 0 .9rem;}
    .notice {background:#fff8e8;border:1px solid #f1dfb8;color:#77551b;
        border-radius:13px;padding:.85rem 1rem;font-size:.84rem;line-height:1.55;}
    .metric {background:#f8fbfd;border:1px solid var(--line);border-radius:14px;
        padding:1rem;min-height:105px;}
    .metric-label {font-size:.78rem;color:var(--muted);}
    .metric-value {font-family:'Manrope',sans-serif;font-size:1.65rem;
        font-weight:800;color:var(--ink);margin-top:.25rem;}
    .metric-note {font-size:.72rem;color:var(--muted);margin-top:.1rem;}
    .top-result {background:var(--teal-soft);border:1px solid #c9e9e3;
        border-radius:16px;padding:1.15rem 1.25rem;}
    .top-result-label {font-size:.75rem;text-transform:uppercase;letter-spacing:.12em;
        color:#28756e;font-weight:800;}
    .top-result-class {font-family:'Manrope',sans-serif;font-size:2rem;
        font-weight:800;color:#145b57;}
    .top-result-score {font-weight:700;color:#28756e;}
    .stButton > button {
        border:0;border-radius:12px;min-height:3.1rem;font-weight:700;
        background:#087f78;color:white;transition:.15s ease;
    }
    .stButton > button:hover {background:#066b65;color:white;box-shadow:0 7px 20px #087f7826;}
    div[data-testid="stFileUploader"] section {
        border:1px dashed #a9bacb;border-radius:13px;background:#f9fcff;
    }
    div[data-testid="stFileUploader"] button {
        background:white;color:var(--ink);border:1px solid var(--line);
        border-radius:10px;
    }
    div[data-testid="stFileUploader"] button:hover {
        background:#ffffff;color:var(--ink);border-color:#a9bacb;
    }
    div[data-testid="stSelectbox"] > div > div {
        border-radius:10px;border-color:var(--line);background:#fff;
    }
    div[data-testid="stSelectbox"] [data-baseweb="select"] {
        color:#000000 !important;
        -webkit-text-fill-color:#000000 !important;
    }
    div[data-testid="stSelectbox"] [data-baseweb="select"] * {
        color:#000000 !important;
        -webkit-text-fill-color:#000000 !important;
    }
    div[data-testid="stSelectbox"] input,
    div[data-testid="stSelectbox"] span,
    div[data-testid="stSelectbox"] div[role="combobox"] {
        color:#000000 !important;
        -webkit-text-fill-color:#000000 !important;
    }
    div[data-baseweb="popover"] [role="option"],
    div[data-baseweb="popover"] [role="option"] * {
        color:#000000 !important;
    }
    div[data-testid="stTable"], div[data-testid="stDataFrame"] {
        border:1px solid var(--line);border-radius:12px;overflow:hidden;
    }
    .footer {color:#8492a5;font-size:.78rem;text-align:center;padding-top:1rem;}
    hr {border-color:var(--line);}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_predictor() -> Predictor:
    return Predictor(Path("models/diagnoai_e2.pt"))


predictor = get_predictor()

SYMPTOM_LABELS = {
    "itching": "Itching",
    "growth": "Has the spot grown?",
    "pain": "Pain or tenderness",
    "appearance_change": "Has its appearance changed?",
    "bleeding": "Bleeding",
    "elevation": "Is the spot raised?",
}
STATE_OPTIONS = ["Yes", "No", "Unknown"]
STATE_MAPPING = {"Unknown": "unknown", "Yes": "yes", "No": "no"}

# Header
st.markdown(
    """
    <div class="hero">
      <div class="hero-kicker">Skin image research tool</div>
      <div class="hero-title">DiagnoAI:Bringing clarity to skin health through AI.</div>
      <div class="hero-copy">
        Add a skin image and the symptoms you have noticed to explore the
        model's research outputs in one simple place.
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)
st.write("")

st.markdown(
    """
    <div class="notice">
      <strong>Important:</strong> DiagnoAI is a research prototype, not a medical
      device. It cannot diagnose or rule out a condition. Scores are not the
      probability that someone has a disease. For any health concern, speak with
      a qualified healthcare professional.
    </div>
    """,
    unsafe_allow_html=True,
)
st.write("")

if not predictor.ready:
    st.error("The application model could not be loaded.")
    st.code(predictor.load_error or "No model details available.")
    st.stop()

# Step 1: image
st.markdown(
    """
    <div class="section-card">
      <div class="step">Step 1</div>
      <div class="section-title">Upload a skin image</div>
      <div class="section-help">
        Use a clear, well-lit image. JPG, JPEG, PNG and WEBP files are supported.
        Maximum file size: <strong>5 MB</strong>.
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)
uploaded = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png", "webp"],
    max_upload_size=5,
    help="Maximum file size is 5 MB.",
    label_visibility="collapsed",
)

image: Image.Image | None = None
if uploaded is not None:
    if uploaded.size > 5 * 1024 * 1024:
        st.error("This image is larger than 5 MB. Please choose a smaller image.")
    else:
        try:
            image = Image.open(uploaded).convert("RGB")
            st.image(image, caption="Image selected", use_container_width=True)
            st.caption(f"Image size: {image.width} × {image.height} pixels")
        except Exception:
            st.error("We couldn't open this image. Please try another file.")

# Step 2: symptoms
st.write("")
st.markdown(
    """
    <div class="section-card">
      <div class="step">Step 2</div>
      <div class="section-title">Tell us what you have noticed</div>
      <div class="section-help">
        Select Yes, No or Unknown for each item. Choose Unknown if you are not sure.
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)
symptom_states: dict[str, str] = {}
symptom_columns = st.columns(2)
for index, symptom in enumerate(SYMPTOM_NAMES):
    with symptom_columns[index % 2]:
        selected = st.selectbox(
            SYMPTOM_LABELS.get(symptom, symptom.replace("_", " ").title()),
            options=STATE_OPTIONS,
            index=None,
            placeholder="Select",
            key=f"symptom_{symptom}",
        )
        symptom_states[symptom] = STATE_MAPPING[selected] if selected is not None else "unknown"

# Step 3: action
st.write("")
st.markdown(
    """
    <div class="section-card">
      <div class="step">Step 3</div>
      <div class="section-title">Review your inputs</div>
      <div class="section-help">
        When your image and symptom responses are ready, run the analysis.
        You can change your answers before submitting.
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)
run_analysis = st.button(
    "Analyze image",
    type="primary",
    use_container_width=True,
    disabled=image is None,
)

if run_analysis and image is not None:
    try:
        with st.spinner("Reviewing the image and your responses..."):
            result = predictor.predict(
                image=image,
                symptom_states=symptom_states,
                top_k=3,
                include_explanations=True,
            )
    except ModelNotReadyError as exc:
        st.error(str(exc))
        st.stop()
    except Exception as exc:
        st.error(f"We couldn't complete the analysis: {exc}")
        st.stop()

    st.write("")
    st.markdown("---")
    st.markdown(
        """
        <div class="step">Your results</div>
        <div class="section-title">Model output summary</div>
        <div class="section-help">
          These are ranked model outputs for research review, not a diagnosis.
        </div>
        """,
        unsafe_allow_html=True,
    )

    predictions = result.get("predictions", [])
    uncertainty = result.get("uncertainty", {})
    if predictions:
        first = predictions[0]
        first_col, list_col = st.columns([1, 1.5], gap="large")
        with first_col:
            st.markdown(
                f"""
                <div class="top-result">
                  <div class="top-result-label">Top-ranked result</div>
                  <div class="top-result-class">{first['label']}</div>
                  <div class="top-result-score">{first['probability']:.1%} model score</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with list_col:
            st.markdown("**Other ranked outputs**")
            for item in predictions:
                st.write(f"**{item['rank']}. {item['label']}** · {item['probability']:.1%}")
                st.progress(float(item["probability"]))

    st.write("")
    c1, c2, c3 = st.columns(3)
    metric_items = [
        ("Top model score", f"{uncertainty.get('top_confidence', 0):.1%}", "Highest model score"),
        ("Top-two difference", f"{uncertainty.get('top_two_margin', 0):.1%}", "Gap between first and second"),
        ("Score distribution", f"{uncertainty.get('normalized_entropy', 0):.2f}", "Normalized entropy"),
    ]
    for col, (label, value, note) in zip((c1, c2, c3), metric_items):
        with col:
            st.markdown(
                f'<div class="metric"><div class="metric-label">{label}</div><div class="metric-value">{value}</div><div class="metric-note">{note}</div></div>',
                unsafe_allow_html=True,
            )

    if uncertainty.get("uncertain"):
        st.warning("The model's uncertainty checks were triggered. Do not rely on the top-ranked result.")
    else:
        st.info("The configured uncertainty checks were not triggered. This does not confirm or rule out a medical condition.")

    st.write("")
    st.markdown(
        """
        <div class="section-title">Visual explanation</div>
        <div class="section-help">
          These tools show how the model responds to the image and symptom inputs.
          They are not clinically validated explanations.
        </div>
        """,
        unsafe_allow_html=True,
    )
    cam = result.get("grad_cam_png_base64")
    if cam:
        st.image(
            base64.b64decode(cam),
            caption="Image highlight for the top-ranked model output. This is not a lesion boundary.",
            use_container_width=True,
        )

    sensitivities = result.get("symptom_sensitivity", [])
    st.markdown("**Symptom sensitivity**")
    st.caption(
        "Shows how the selected class score changes when a reported Yes response is changed to No, one at a time. This is model sensitivity, not cause and effect."
    )
    if sensitivities:
        st.dataframe(
            [
                {
                    "Symptom": item["symptom"].replace("_", " ").title(),
                    "Change in model score": item["score_drop"],
                }
                for item in sensitivities
            ],
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.write("No symptoms marked Yes were available for this comparison.")

    validation = result.get("validation_metrics", {})
    if validation:
        st.write("")
        st.markdown("**Model validation summary**")
        st.caption("These figures come from the validation dataset, not this uploaded image or the held-out test set.")
        vm1, vm2, vm3, vm4 = st.columns(4)
        vals = [
            ("Accuracy", validation.get("accuracy")),
            ("Macro-F1", validation.get("macro_f1")),
            ("NLL", validation.get("negative_log_likelihood")),
            ("ECE", validation.get("top_label_ece_10_bins")),
        ]
        for col, (label, value) in zip((vm1, vm2, vm3, vm4), vals):
            with col:
                st.metric(label, f"{value:.4f}" if isinstance(value, (int, float)) else "N/A")

    st.write("")
    st.markdown(
        """
        <div class="notice">
          <strong>Please remember:</strong> A reassuring-looking score or heatmap
          cannot rule out a serious condition. If a spot is new, changing, bleeding,
          painful or concerning, seek advice from a qualified healthcare professional.
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown(
    '<div class="footer">DiagnoAI · Research prototype · Not for medical diagnosis</div>',
    unsafe_allow_html=True,
)
