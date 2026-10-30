"""Streamlit interface for the trained customer support intent classifier."""
import json
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "customer_support_intent_model.keras"
METADATA_PATH = BASE_DIR / "models" / "customer_support_metadata.json"

st.set_page_config(page_title="Customer Support Chatbot", page_icon="💬", layout="centered")

@st.cache_resource
def load_artifacts():
    if not MODEL_PATH.exists() or not METADATA_PATH.exists():
        return None, None
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(METADATA_PATH, "r", encoding="utf-8") as file:
        metadata = json.load(file)
    return model, metadata

model, metadata = load_artifacts()

st.title("💬 Customer Support Chatbot")
st.caption("Deep Learning + Natural Language Processing (NLP)")
st.markdown("Ask a question about orders, delivery, refunds, payments, accounts, or customer support.")

if model is None or metadata is None:
    st.warning("The trained model files were not found.")
    st.markdown("Run **python train_model.py** first. It will create `customer_support_intent_model.keras` and `customer_support_metadata.json` in this project folder.")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role": "assistant",
        "content": "Hello! I'm your customer support assistant. How can I help you today?",
    }]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("details"):
            with st.expander("Prediction details"):
                st.write(message["details"])

prompt = st.chat_input("Type your customer-support question...")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    probabilities = model.predict(np.array([prompt], dtype=object), verbose=0)[0]
    predicted_id = int(np.argmax(probabilities))
    intent_names = metadata["intent_names"]
    intent = intent_names[predicted_id]
    confidence = float(probabilities[predicted_id])
    threshold = float(metadata.get("confidence_threshold", 0.60))

    if confidence < threshold:
        response = metadata.get("fallback_response", "I'm not sure I understood. Could you rephrase your question?")
    else:
        response = metadata.get("response_templates", {}).get(
            intent, metadata.get("fallback_response", "Please contact official customer support for help.")
        )

    details = {
        "Predicted intent": intent,
        "Confidence": f"{confidence:.1%}",
        "Fallback used": confidence < threshold,
    }
    with st.chat_message("assistant"):
        st.markdown(response)
        with st.expander("Prediction details"):
            st.write(details)
    st.session_state.messages.append({"role": "assistant", "content": response, "details": details})

with st.sidebar:
    st.header("About this project")
    st.write("This application predicts the intent of a customer message using a neural-network text classifier.")
    st.write(f"**Supported intents:** {len(metadata['intent_names'])}")
    st.write(f"**Confidence threshold:** {metadata.get('confidence_threshold', 0.60):.0%}")
    st.info("This is an academic demonstration. It does not access real orders, payments, refunds, or customer accounts.")
    if st.button("Clear conversation"):
        st.session_state.messages = [{
            "role": "assistant",
            "content": "Hello! I'm your customer support assistant. How can I help you today?",
        }]
        st.rerun()
