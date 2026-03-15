import streamlit as st
from model_loader import model_service


st.set_page_config(
    page_title="EDOS Classifier",
    page_icon="🧠",
    layout="centered"
)

st.markdown(
    """
    <style>
    :root {
        --cream-bg: #f7f1e3;
        --cream-surface: #fffaf0;
        --cream-border: #d8c7a1;
        --cream-accent: #8c6a43;
        --cream-accent-soft: #eadfc8;
        --cream-text: #3f3122;
    }

    .stApp {
        background: linear-gradient(180deg, #f9f3e7 0%, #f3ead7 100%);
        color: var(--cream-text);
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    .stApp,
    .stApp p,
    .stApp label,
    .stApp div {
        color: var(--cream-text);
    }

    .stApp h1,
    .stApp h2,
    .stApp h3,
    .stApp [data-testid="stMarkdownContainer"] h1,
    .stApp [data-testid="stMarkdownContainer"] h2,
    .stApp [data-testid="stMarkdownContainer"] h3 {
        color: var(--cream-text) !important;
    }

    [data-testid="stTextArea"] textarea {
        background-color: var(--cream-surface);
        border: 1px solid var(--cream-border);
        border-radius: 12px;
        color: var(--cream-text);
    }

    [data-testid="stTextArea"] textarea:focus {
        border-color: var(--cream-accent);
        box-shadow: 0 0 0 1px var(--cream-accent);
    }

    .stButton > button {
        background-color: var(--cream-accent);
        color: #fffaf0;
        border: 1px solid var(--cream-accent);
        border-radius: 999px;
        padding: 0.55rem 1.4rem;
        font-weight: 600;
    }

    .stButton > button:hover {
        background-color: #765636;
        border-color: #765636;
    }

    [data-testid="stAlert"] {
        background-color: var(--cream-surface);
        border: 1px solid var(--cream-border);
        color: var(--cream-text);
    }

    [data-testid="stProgressBar"] > div > div {
        background-color: var(--cream-accent);
    }
    </style>
    """,
    unsafe_allow_html=True
)


@st.cache_resource
def get_model_service():
    model_service.load()
    return model_service


service = get_model_service()

st.title("EDOS Sexism Classifier")
st.write(
    "Most social media filters are like a 'black box'—they block comments "
    "but never tell you why. This site changes that. We don't just find sexist "
    "comments; we explain them. Our AI identifies 11 specific types of sexism "
    "and actually highlights the exact words that caused the red flag. It's "
    "like having an AI that 'shows its work,' making digital moderation fairer, "
    "clearer, and easier to trust."
)

text = st.text_area(
    "Input text",
    height=180,
    placeholder="Type or paste text here..."
)

predict_btn = st.button("Predict")

if predict_btn:
    if not text.strip():
        st.warning("Please enter some text first.")
    else:
        with st.spinner("Running prediction..."):
            result = service.predict(text)

        st.success("Prediction complete")

        st.subheader("Task A")
        st.write(f"**Prediction:** {result['task_a']['label']}")
        for i, p in enumerate(result["task_a"]["probs"]):
            st.write(f"Class {i}: {p:.4f}")
            st.progress(float(p))

        st.subheader("Task B")
        st.write(f"**Prediction:** {result['task_b']['label']}")
        for i, p in enumerate(result["task_b"]["probs"]):
            st.write(f"Class {i}: {p:.4f}")
            st.progress(float(p))

        st.subheader("Task C")
        st.write(f"**Prediction:** {result['task_c']['label']}")
        st.write(f"**Short label:** {result['task_c']['short_label']}")
        for i, p in enumerate(result["task_c"]["probs"]):
            st.write(f"Class {i}: {p:.4f}")
            st.progress(float(p))
