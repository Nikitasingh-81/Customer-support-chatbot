"""
Train the Customer Support Intent Classification model.

Expected input files (place them beside this script):
- Bitext_Sample_Customer_Service_Training_Dataset.csv
- Bitext_Sample_Customer_Service_Validation_Dataset.csv
- Bitext_Sample_Customer_Service_Testing_Dataset.csv

Outputs:
- models/customer_support_intent_model.keras
- models/customer_support_metadata.json

The model accepts raw customer-message strings. TextVectorization is included
inside the saved Keras model, so app.py can pass a message directly to model.predict().
"""

import json
import random
import re
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder


# ---------------------------------------------------------------------
# 1. Configuration
# ---------------------------------------------------------------------

SEED = 42
MAX_TOKENS = 20_000
SEQUENCE_LENGTH = 40
EMBEDDING_DIM = 64
BATCH_SIZE = 32
EPOCHS = 20
CONFIDENCE_THRESHOLD = 0.60

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR
MODEL_DIR = BASE_DIR / "models"

TRAIN_FILE = DATA_DIR / "Bitext_Sample_Customer_Service_Training_Dataset.csv"
VALIDATION_FILE = DATA_DIR / "Bitext_Sample_Customer_Service_Validation_Dataset.csv"
TEST_FILE = DATA_DIR / "Bitext_Sample_Customer_Service_Testing_Dataset.csv"

MODEL_PATH = MODEL_DIR / "customer_support_intent_model.keras"
METADATA_PATH = MODEL_DIR / "customer_support_metadata.json"


# ---------------------------------------------------------------------
# 2. Reproducibility and setup
# ---------------------------------------------------------------------

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

MODEL_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# 3. Load the three supplied datasets
# ---------------------------------------------------------------------

def load_dataset(path: Path, split_name: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find {path.name}. Put this CSV in the same folder "
            f"as train_model.py. Expected location: {path}"
        )

    df = pd.read_csv(path)

    required_columns = {"utterance", "intent"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(
            f"{path.name} is missing required columns: {sorted(missing_columns)}. "
            f"Found columns: {list(df.columns)}"
        )

    # Keep only rows with usable text and intent labels.
    df = df[["utterance", "intent"]].copy()
    df["utterance"] = df["utterance"].fillna("").astype(str).str.strip()
    df["intent"] = df["intent"].fillna("").astype(str).str.strip()
    df = df[(df["utterance"] != "") & (df["intent"] != "")].copy()

    # Remove exact duplicate text/label pairs within this split.
    before = len(df)
    df = df.drop_duplicates(subset=["utterance", "intent"]).reset_index(drop=True)
    removed = before - len(df)

    print(f"{split_name}: {len(df):,} usable rows; removed {removed:,} exact duplicates.")
    return df


train_df = load_dataset(TRAIN_FILE, "Training")
validation_df = load_dataset(VALIDATION_FILE, "Validation")
test_df = load_dataset(TEST_FILE, "Testing")


# ---------------------------------------------------------------------
# 4. Check label consistency across the supplied splits
# ---------------------------------------------------------------------

train_intents = set(train_df["intent"].unique())
validation_unknown = sorted(set(validation_df["intent"].unique()) - train_intents)
test_unknown = sorted(set(test_df["intent"].unique()) - train_intents)

if validation_unknown or test_unknown:
    raise ValueError(
        "Some validation/testing intents do not occur in the training dataset. "
        "A classifier cannot learn classes it never sees.\n"
        f"Validation-only intents: {validation_unknown}\n"
        f"Testing-only intents: {test_unknown}\n"
        "Check the source CSV files and their intent labels before training."
    )

print(f"Number of intent classes: {len(train_intents)}")


# ---------------------------------------------------------------------
# 5. Encode intent labels consistently
# ---------------------------------------------------------------------

label_encoder = LabelEncoder()
label_encoder.fit(train_df["intent"])

y_train = label_encoder.transform(train_df["intent"])
y_validation = label_encoder.transform(validation_df["intent"])
y_test = label_encoder.transform(test_df["intent"])

intent_names = label_encoder.classes_.tolist()
num_classes = len(intent_names)

X_train = train_df["utterance"].astype(str).to_numpy()
X_validation = validation_df["utterance"].astype(str).to_numpy()
X_test = test_df["utterance"].astype(str).to_numpy()


# ---------------------------------------------------------------------
# 6. Create a text vectorization layer
#    It is adapted ONLY on training utterances.
# ---------------------------------------------------------------------

vectorizer = tf.keras.layers.TextVectorization(
    max_tokens=MAX_TOKENS,
    output_mode="int",
    output_sequence_length=SEQUENCE_LENGTH,
    standardize="lower_and_strip_punctuation",
)

vectorizer.adapt(tf.data.Dataset.from_tensor_slices(X_train).batch(256))


# ---------------------------------------------------------------------
# 7. Build the neural-network classifier
#    TextVectorization is part of the model, so app.py can pass raw text.
# ---------------------------------------------------------------------

text_input = tf.keras.Input(shape=(), dtype=tf.string, name="customer_message")

x = vectorizer(text_input)
x = tf.keras.layers.Embedding(
    input_dim=MAX_TOKENS,
    output_dim=EMBEDDING_DIM,
    name="word_embedding",
)(x)
x = tf.keras.layers.GlobalAveragePooling1D()(x)
x = tf.keras.layers.Dense(128, activation="relu")(x)
x = tf.keras.layers.Dropout(0.30)(x)
x = tf.keras.layers.Dense(64, activation="relu")(x)
x = tf.keras.layers.Dropout(0.20)(x)

output = tf.keras.layers.Dense(
    num_classes,
    activation="softmax",
    name="intent_probabilities",
)(x)

model = tf.keras.Model(inputs=text_input, outputs=output, name="customer_support_intent_model")

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()


# ---------------------------------------------------------------------
# 8. Train using training data and monitor validation data
# ---------------------------------------------------------------------

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True,
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        min_lr=1e-6,
    ),
]

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_validation, y_validation),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
    verbose=1,
)


# ---------------------------------------------------------------------
# 9. Evaluate once on the held-out testing dataset
# ---------------------------------------------------------------------

test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest loss: {test_loss:.4f}")
print(f"Test accuracy: {test_accuracy:.4f}")

test_probabilities = model.predict(X_test, batch_size=BATCH_SIZE, verbose=0)
test_predictions = np.argmax(test_probabilities, axis=1)

print("\nClassification report:")
print(
    classification_report(
        y_test,
        test_predictions,
        labels=np.arange(num_classes),
        target_names=intent_names,
        zero_division=0,
    )
)

print("\nConfusion matrix:")
print(confusion_matrix(y_test, test_predictions))


# ---------------------------------------------------------------------
# 10. Create simple response templates
#     These are example text replies, not live order/refund operations.
# ---------------------------------------------------------------------

def readable_intent(intent: str) -> str:
    """Turn labels such as order_status into readable words."""
    return re.sub(r"[_\-]+", " ", str(intent)).strip()


response_templates = {
    intent: (
        f"I understand that your question is about {readable_intent(intent)}. "
        "I can provide general guidance, but I cannot access or change your "
        "orders, payments, refunds, or account. Please contact the company's "
        "official support team for account-specific assistance."
    )
    for intent in intent_names
}

fallback_response = (
    "I'm not fully sure I understood your request. Could you rephrase it or "
    "provide a little more detail? For account-specific help, please contact "
    "the company's official customer-support team."
)


# ---------------------------------------------------------------------
# 11. Save the trained model and metadata for Streamlit
# ---------------------------------------------------------------------

model.save(MODEL_PATH)

metadata = {
    "intent_names": intent_names,
    "intent_to_id": {
        intent: int(index)
        for index, intent in enumerate(intent_names)
    },
    "id_to_intent": {
        str(index): intent
        for index, intent in enumerate(intent_names)
    },
    "response_templates": response_templates,
    "fallback_response": fallback_response,
    "confidence_threshold": CONFIDENCE_THRESHOLD,
    "model_type": "Keras text intent classifier",
    "text_vectorization_included_in_model": True,
    "sequence_length": SEQUENCE_LENGTH,
    "max_tokens": MAX_TOKENS,
    "embedding_dimension": EMBEDDING_DIM,
    "test_loss": float(test_loss),
    "test_accuracy": float(test_accuracy),
}

with open(METADATA_PATH, "w", encoding="utf-8") as file:
    json.dump(metadata, file, indent=4, ensure_ascii=False)

print("\nTraining and saving completed.")
print(f"Model file:    {MODEL_PATH}")
print(f"Metadata file: {METADATA_PATH}")


# ---------------------------------------------------------------------
# 12. Reload the saved model and test one sample
# ---------------------------------------------------------------------

reloaded_model = tf.keras.models.load_model(MODEL_PATH)

sample_message = "Where is my order?"
sample_probabilities = reloaded_model.predict(
    np.array([sample_message], dtype=object),
    verbose=0,
)[0]

sample_id = int(np.argmax(sample_probabilities))
sample_intent = intent_names[sample_id]
sample_confidence = float(sample_probabilities[sample_id])

print("\nReload test:")
print("Message:", sample_message)
print("Predicted intent:", sample_intent)
print(f"Confidence: {sample_confidence:.2%}")
print("Response:", response_templates[sample_intent])

print("\nBoth required files are ready in the models folder.")
