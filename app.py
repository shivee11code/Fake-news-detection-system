import streamlit as st
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(
    page_title="Fake News Detection System",
    page_icon="📰",
    layout="centered"
)

@st.cache_resource
def load_resources():
    model = joblib.load("model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
    cm = joblib.load("confusion_matrix.pkl")
    return model, vectorizer, cm

model, vectorizer, cm = load_resources()

st.title("📰 Fake News Detection System")
st.markdown("Classify news articles as **Fake** or **Real** using Machine Learning.")

st.divider()

article = st.text_area("Enter News Article Text", height=200)

if st.button("Predict", type="primary"):
    if article.strip() == "":
        st.warning("Please enter article text.")
    else:
        transformed = vectorizer.transform([article])
        prediction = model.predict(transformed)[0]

        try:
            probs = model.predict_proba(transformed)[0]
            confidence = np.max(probs) * 100
        except:
            confidence = None

        if prediction == 1:
            st.error("🚨 Fake News Detected")
        else:
            st.success("✅ Real News")

        if confidence:
            st.progress(int(confidence))
            st.write(f"Confidence: {confidence:.2f}%")

st.divider()

st.subheader("Confusion Matrix (Test Data)")

fig, ax = plt.subplots()
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Real", "Fake"],
    yticklabels=["Real", "Fake"],
    ax=ax
)

st.pyplot(fig)
