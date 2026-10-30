# Customer Support Chatbot Using Deep Learning and NLP

## Project overview
This project classifies a customer's message into an intent (for example, tracking an order, requesting a refund, or recovering a password) using a TensorFlow/Keras neural network. A Streamlit web interface lets a user test the trained model in a browser.

**Academic scope:** The application predicts intent and displays a predefined guidance message. It does not connect to a real company's order system, process payments, issue refunds, or access customer accounts.

## Dataset
The `data/` directory contains the three supplied Bitext sample customer-service CSV files. Each file should contain these columns:

- `utterance`: the customer's text message
- `intent`: the intent label to predict
- `category`: the broader service category
- `tags`: dataset tags

The supplied splits are retained separately: training data is used to fit the text vocabulary and model; validation data is used during training; test data is reserved for final evaluation.

## Files
- `train_model.py` — data checks, EDA plots, text vectorization, model training, evaluation, and saving artifacts.
- `app.py` — Streamlit chatbot interface.
- `requirements.txt` — Python package dependencies.
- `data/` — supplied training, validation, and testing CSV files.
- `outputs/` — generated plots, metrics, classification report, confusion matrix, and misclassified examples (created after training).

## Run locally (recommended for a live demonstration)
Use Python 3.10 or 3.11 in a virtual environment if possible.

```bash
python -m venv .venv
```

Activate the environment:
- Windows PowerShell: `.venv\\Scripts\\Activate.ps1`
- macOS/Linux: `source .venv/bin/activate`

Install packages:

```bash
pip install -r requirements.txt
```

Train and evaluate the model:

```bash
python train_model.py
```

Training creates:
- `customer_support_intent_model.keras`
- `customer_support_metadata.json`
- evaluation outputs in `outputs/`

Start the web application:

```bash
streamlit run app.py
```

A browser tab should open with the chatbot. If it does not, open the local URL printed in the terminal (usually `http://localhost:8501`).

## Run in Google Colab
1. Upload the project files or upload the ZIP and extract it in Colab.
2. Ensure the three CSVs are in the `data/` folder.
3. Install packages with `!pip install -r requirements.txt`.
4. Run `!python train_model.py` (training may take time).
5. Streamlit is not automatically public just because the training code ran in Colab. For a reliable presentation, run the app locally or deploy it through a supported Streamlit hosting workflow.

## Model design
`TextVectorization → Embedding → GlobalAveragePooling1D → Dense(ReLU) → Dropout → Dense(Softmax)`

- TextVectorization converts raw text to token IDs and learns its vocabulary from training messages only.
- Embedding learns a dense representation of tokens.
- GlobalAveragePooling1D aggregates token representations into one message vector.
- Dense/ReLU learns nonlinear features.
- Dropout reduces overfitting risk during training.
- Softmax returns a probability distribution over the intent labels.

## Results and reporting
Do not write performance numbers in the report until `train_model.py` has completed. Use `outputs/evaluation_metrics.json`, `outputs/classification_report.txt`, and the saved plots to report the actual run. Model scores can vary by environment and training behavior.

## Suggested demonstration messages
- `Where is my package?`
- `I need a refund for my order.`
- `I forgot my password.`
- `I want to cancel my order.`
- `I want to speak to a human agent.`

Actual predictions depend on the trained model. A low-confidence prediction triggers the fallback response.

## Limitations
- Intent classification is not the same as a generative chatbot; responses are selected from predefined templates.
- The model may misclassify messages that are ambiguous, very different from the training examples, or contain multiple requests.
- The confidence score is the model's softmax score and is not a guarantee of correctness.
- No live company systems or customer records are connected.


## GitHub folder layout

- `app.py` — Streamlit application.
- `requirements.txt` — Python dependencies.
- `notebooks/Customer_Support_Chatbot.ipynb` — Google Colab notebook.
- `models/` — add the trained `.keras` model and generated metadata JSON here.
- `screenshots/` — add genuine screenshots from your run here.

After running the notebook, download `customer_support_intent_model.keras` and
`customer_support_metadata.json` and place both inside `models/`. The Streamlit
app expects them at those exact paths.
