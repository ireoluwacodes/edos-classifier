import streamlit as st
from model_loader import model_service


st.set_page_config(
    page_title="EDOS Classifier",
    page_icon="🧠",
    layout="centered"
)


@st.cache_resource
def get_model_service():
    model_service.load()
    return model_service


service = get_model_service()

st.title("EDOS Sexism Classifier")
st.write("Enter text below to classify it across Task A, Task B, and Task C.")

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