"""
Train the Customer Support Chatbot using the project's original architecture.

Run from the repository root:
    python train_model.py

The three CSV files are read from data/. The trained model and metadata are
saved to models/. This script mirrors the original Colab notebook's
TextVectorization -> Embedding -> GlobalAveragePooling1D -> Dense -> Dropout
-> Softmax architecture and response-template dictionary.
"""
from pathlib import Path
import json
import os
import random

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "outputs"

TRAIN_FILE = DATA_DIR / "Bitext_Sample_Customer_Service_Training_Dataset.csv"
VAL_FILE = DATA_DIR / "Bitext_Sample_Customer_Service_Validation_Dataset.csv"
TEST_FILE = DATA_DIR / "Bitext_Sample_Customer_Service_Testing_Dataset.csv"

MODEL_PATH = MODEL_DIR / "customer_support_intent_model.keras"
METADATA_PATH = MODEL_DIR / "customer_support_metadata.json"

SEED = 42
MAX_TOKENS = 12000
SEQUENCE_LENGTH = 40
EMBEDDING_DIM = 64
BATCH_SIZE = 32
EPOCHS = 20
CONFIDENCE_THRESHOLD = 0.60

random.seed(SEED)
np.random.seed(SEED)
tf.keras.utils.set_random_seed(SEED)

for folder in (MODEL_DIR, OUTPUT_DIR):
    folder.mkdir(parents=True, exist_ok=True)

def load_split(path: Path, split_name: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Ensure the repository's data/ folder contains "
            "all three Bitext CSV files."
        )
    df = pd.read_csv(path)
    required = {"utterance", "intent", "category", "tags"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{split_name} dataset missing columns: {sorted(missing)}")
    df = df.copy()
    df["utterance"] = df["utterance"].fillna("").astype(str).str.strip()
    df["intent"] = df["intent"].fillna("").astype(str).str.strip()
    df = df[(df["utterance"] != "") & (df["intent"] != "")]
    # Keep the supplied train/validation/test splits unchanged.
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"{split_name}: {df.shape[0]} rows")
    return df

print("Loading datasets...")
train_df = load_split(TRAIN_FILE, "Training")
val_df = load_split(VAL_FILE, "Validation")
test_df = load_split(TEST_FILE, "Testing")

train_intents = set(train_df["intent"].unique())
val_intents = set(val_df["intent"].unique())
test_intents = set(test_df["intent"].unique())
if train_intents != val_intents or train_intents != test_intents:
    raise ValueError(
        "Intent labels differ across supplied splits. "
        f"Validation-only/missing: {sorted(val_intents ^ train_intents)}; "
        f"Testing-only/missing: {sorted(test_intents ^ train_intents)}"
    )

print(f"Number of intent classes: {len(train_intents)}")
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(train_df["intent"])
y_val = label_encoder.transform(val_df["intent"])
y_test = label_encoder.transform(test_df["intent"])
intent_names = label_encoder.classes_.tolist()
num_classes = len(intent_names)

X_train = train_df["utterance"].astype(str).to_numpy()
X_val = val_df["utterance"].astype(str).to_numpy()
X_test = test_df["utterance"].astype(str).to_numpy()

print("Building text vectorizer (adapted on training messages only)...")
vectorizer = tf.keras.layers.TextVectorization(
    max_tokens=MAX_TOKENS,
    output_mode="int",
    output_sequence_length=SEQUENCE_LENGTH,
    standardize="lower_and_strip_punctuation",
    name="text_vectorization",
)
vectorizer.adapt(tf.data.Dataset.from_tensor_slices(X_train).batch(256))

inputs = tf.keras.Input(shape=(), dtype=tf.string, name="customer_message")
x = vectorizer(inputs)
x = tf.keras.layers.Embedding(
    MAX_TOKENS, EMBEDDING_DIM, name="word_embedding"
)(x)
x = tf.keras.layers.GlobalAveragePooling1D(
    name="global_average_pooling"
)(x)
x = tf.keras.layers.Dense(64, activation="relu", name="hidden_dense")(x)
x = tf.keras.layers.Dropout(0.30, name="dropout")(x)
outputs = tf.keras.layers.Dense(
    num_classes, activation="softmax", name="intent_probabilities"
)(x)

model = tf.keras.Model(
    inputs=inputs,
    outputs=outputs,
    name="customer_support_intent_classifier",
)
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
model.summary()

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=3, restore_best_weights=True, verbose=1
    ),
    tf.keras.callbacks.ModelCheckpoint(
        str(MODEL_PATH), monitor="val_loss", save_best_only=True, verbose=1
    ),
]
print("Training model...")
history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
    verbose=1,
)

# Load the best validation-loss checkpoint and evaluate on test data once.
model = tf.keras.models.load_model(MODEL_PATH)
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
test_probabilities = model.predict(X_test, batch_size=BATCH_SIZE, verbose=0)
y_pred = np.argmax(test_probabilities, axis=1)

report = classification_report(
    y_test,
    y_pred,
    labels=np.arange(num_classes),
    target_names=intent_names,
    zero_division=0,
)
(OUTPUT_DIR / "classification_report.txt").write_text(report, encoding="utf-8")
cm = confusion_matrix(y_test, y_pred, labels=np.arange(num_classes))
pd.DataFrame(cm, index=intent_names, columns=intent_names).to_csv(
    OUTPUT_DIR / "confusion_matrix.csv", index=True
)
error_df = pd.DataFrame({
    "utterance": X_test,
    "actual_intent": [intent_names[i] for i in y_test],
    "predicted_intent": [intent_names[i] for i in y_pred],
    "confidence": np.max(test_probabilities, axis=1),
})
error_df[error_df["actual_intent"] != error_df["predicted_intent"]].to_csv(
    OUTPUT_DIR / "misclassified_examples.csv", index=False
)

RESPONSE_TEMPLATES = {
    "cancel_order": "I can guide you on order cancellation. Please check your order details and the seller's cancellation policy.",
    "change_order": "Please review your order status to see whether changes are still available.",
    "change_shipping_address": "Please check your order status and account options as soon as possible to see whether the shipping address can be changed.",
    "check_cancellation_fee": "Cancellation fees depend on the applicable order terms. Please review the fee information for your order.",
    "check_invoice": "You can usually find invoice details in the order or billing section of your account.",
    "get_invoice": "Please check your order history or billing section for invoice download options.",
    "payment_issue": "I can help with general payment troubleshooting. Check the payment status and never share full card details or passwords.",
    "check_payment_methods": "Available payment methods are normally shown during checkout or in payment settings.",
    "get_refund": "For a refund request, review the seller's refund process and the status of your order.",
    "track_refund": "Refund timing depends on the seller and payment provider. Check your account or contact official support for an update.",
    "check_refund_policy": "Please review the seller's refund policy for eligibility, timelines, and conditions.",
    "track_order": "You can usually track an order from your order history or the tracking link in your confirmation message.",
    "delivery_period": "Delivery estimates depend on the destination, shipping method, and seller. Check the estimated date in your order details.",
    "delivery_options": "Available delivery options are generally shown during checkout.",
    "place_order": "To place an order, add the desired items to your cart and follow the checkout steps.",
    "set_up_shipping_address": "You can usually add a shipping address in account settings or during checkout.",
    "create_account": "Use the sign-up option and follow the instructions to create an account.",
    "edit_account": "Account details can generally be updated from your profile or account settings.",
    "delete_account": "Please review account settings or contact the service's official support team to request account deletion.",
    "recover_password": "Use the 'Forgot password' option and follow the password-reset instructions.",
    "registration_problems": "Check that required fields are complete and your email address is entered correctly. If the issue continues, contact support.",
    "switch_account": "If the service supports multiple accounts, sign out and sign in with the intended account.",
    "newsletter_subscription": "Newsletter subscription settings are usually available in your profile or at the bottom of newsletter emails.",
    "review": "You can usually submit feedback or a review from the product or order page.",
    "complaint": "I'm sorry you're experiencing a problem. Please describe the issue without including sensitive personal or payment information.",
    "contact_customer_service": "Please use the service's official customer-support channel for help with your specific case.",
    "contact_human_agent": "For help from a person, please use the service's official contact or live-chat option.",
}
FALLBACK_RESPONSE = (
    "I'm not fully sure I understood that. Could you rephrase your question "
    "or contact the service's official support team?"
)
for intent in intent_names:
    RESPONSE_TEMPLATES.setdefault(
        intent,
        "I can provide general guidance, but please contact official customer support for help with this specific request.",
    )

metadata = {
    "project_title": "Customer Support Chatbot Using Deep Learning and NLP",
    "intent_names": intent_names,
    "response_templates": RESPONSE_TEMPLATES,
    "fallback_response": FALLBACK_RESPONSE,
    "confidence_threshold": CONFIDENCE_THRESHOLD,
    "max_tokens": MAX_TOKENS,
    "sequence_length": SEQUENCE_LENGTH,
    "embedding_dim": EMBEDDING_DIM,
    "seed": SEED,
}
with METADATA_PATH.open("w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2, ensure_ascii=False)

metrics = {
    "test_accuracy": float(test_accuracy),
    "test_loss": float(test_loss),
    "training_rows": int(len(train_df)),
    "validation_rows": int(len(val_df)),
    "testing_rows": int(len(test_df)),
    "number_of_intents": int(len(intent_names)),
    "epochs_completed": int(len(history.history["loss"])),
}
with (OUTPUT_DIR / "evaluation_metrics.json").open("w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2)

print("\nTraining complete.")
print(f"Model:    {MODEL_PATH}")
print(f"Metadata: {METADATA_PATH}")
print(f"Test accuracy: {test_accuracy:.4f}")
print(f"Test loss: {test_loss:.4f}")
print(f"Evaluation outputs: {OUTPUT_DIR}")
