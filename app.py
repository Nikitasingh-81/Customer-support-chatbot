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
    missing = [str(p.relative_to(BASE_DIR)) for p in (MODEL_PATH, METADATA_PATH) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Required trained files are missing: " + ", ".join(missing)
            + ". Train the model and add both files to the repository's models/ folder."
        )
    model = tf.keras.models.load_model(MODEL_PATH)
    with METADATA_PATH.open("r", encoding="utf-8") as f:
        metadata = json.load(f)
    required = {"intent_names", "response_templates", "fallback_response"}
    missing_keys = required - set(metadata)
    if missing_keys:
        raise ValueError(f"Metadata is missing keys: {sorted(missing_keys)}")
    return model, metadata

try:
    model, metadata = load_artifacts()
except Exception as exc:
    st.title("💬 Customer Support Chatbot")
    st.caption("Deep Learning + Natural Language Processing (NLP)")
    st.error("The trained model files could not be loaded.")
    st.code(str(exc))
    st.info(
        "Place customer_support_intent_model.keras and "
        "customer_support_metadata.json inside the repository's models/ folder."
    )
    st.stop()

st.title("💬 Customer Support Chatbot")
st.caption("Deep Learning + Natural Language Processing (NLP)")
st.write("Ask a question about orders, delivery, refunds, payments, accounts, or customer support.")
st.info(
    "Academic prototype: this app predicts customer-message intent and returns predefined guidance. "
    "It does not access real orders, payments, refunds, shipping systems, or customer accounts."
)

if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role": "assistant",
        "content": "Hello! I'm your customer support assistant. How can I help you today?",
    }]

# Render previous messages and prediction details without st.json.
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        details = message.get("details")
        if details:
            with st.expander("Prediction details"):
                for key, value in details.items():
                    st.write(f"**{key}:** {value}")

prompt = st.chat_input("Type your customer-support question...")
if prompt:
    prompt = prompt.strip()
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        probabilities = model.predict(np.asarray([prompt], dtype=object), verbose=0)[0]
        predicted_id = int(np.argmax(probabilities))
        intent_names = metadata["intent_names"]

        if predicted_id >= len(intent_names):
            st.error("Model output does not match intent metadata. Retrain and upload matching files.")
            st.stop()

        intent = intent_names[predicted_id]
        confidence = float(probabilities[predicted_id])
        threshold = float(metadata.get("confidence_threshold", 0.60))
        fallback_used = confidence < threshold

        response = (
            metadata["fallback_response"]
            if fallback_used
            else metadata["response_templates"].get(intent, metadata["fallback_response"])
        )

        details = {
            "Predicted intent": intent,
            "Confidence": f"{confidence:.2%}",
            "Fallback used": "Yes" if fallback_used else "No",
        }

        with st.chat_message("assistant"):
            st.markdown(response)
            with st.expander("Prediction details"):
                st.write("**Predicted intent:**", intent)
                st.write("**Confidence:**", f"{confidence:.2%}")
                st.write("**Fallback used:**", "Yes" if fallback_used else "No")

        st.session_state.messages.append({
            "role": "assistant",
            "content": response,
            "details": details,
        })

with st.sidebar:
    st.header("About this project")
    st.write("A TensorFlow/Keras neural network classifies customer messages into support intents.")
    st.write(f"Supported intents: {len(metadata['intent_names'])}")
    st.write(f"Confidence threshold: {float(metadata.get('confidence_threshold', 0.60)):.0%}")
    if st.button("Clear conversation"):
        st.session_state.messages = [{
            "role": "assistant",
            "content": "Hello! I'm your customer support assistant. How can I help you today?",
        }]
        st.rerun()
