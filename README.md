# AI Prompt Firewall

**An ML-powered prompt injection detection and filtering gateway for LLM applications.**

Large language models (LLMs) accept natural-language input, but malicious prompts can attempt to override intended instructions, expose hidden system prompts, or manipulate how a model responds. These attacks can be disguised as ordinary messages, embedded within other text, or phrased indirectly, making them challenging to identify reliably.

AI Prompt Firewall addresses this problem by combining a machine learning text classifier with rule-based detection to analyze incoming prompts before they reach a language model. It assigns a risk score, classifies each prompt as benign or suspicious, and uses a FastAPI gateway to block prompts identified as suspicious. Prompts classified as benign can be forwarded to a locally running LLM. By combining machine learning, text normalization, adversarial testing, and API security, this project demonstrates a practical approach to adding a screening layer to LLM-powered applications.

## Features

- **Machine-learning detection:** Classifies prompts as benign or suspicious.
- **Rule-based detection:** Detects selected prompt-injection patterns.
- **Text normalization:** Standardizes input before classification.
- **FastAPI gateway:** Blocks suspicious prompts and forwards accepted prompts.
- **Local LLM integration:** Connects to Ollama running Qwen 2.5 3B.
- **Web chatbot:** Provides a browser interface for testing prompts.
- **Automated testing:** Tests the detector, API, gateway, text normalization, and adversarial cases.
- **Evaluation tools:** Includes scripts for model evaluation and error analysis.

## How It Works

1. A user submits a prompt through the web interface or API.
2. The detector analyzes the prompt using configured rules and the trained classifier.
3. The detector returns a classification and risk score.
4. The gateway blocks prompts classified as suspicious with HTTP `403`.
5. Accepted prompts are forwarded to the local Ollama model.
6. The application returns the detection result and, for accepted prompts, the LLM response.

## Technology Stack

- Python 3.13
- FastAPI
- Uvicorn
- scikit-learn
- joblib
- pandas and NumPy
- HTTPX
- pytest
- Ollama
- Qwen 2.5 3B

## Requirements

Before running the application, install:

1. Python 3.13 or a compatible version.
2. Git for Windows.
3. Ollama for Windows.
4. The `qwen2.5:3b` model through Ollama.

Python packages are listed in `requirements.txt`. Ollama and its language model must be installed separately.

## Installation and Setup

### Step 1 — Clone the repository

```bash
git clone https://github.com/Anushapeddapalli/AI-prompt-firewall.git
cd AI-prompt-firewall
```

### Step 2 — Create a virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### Step 3 — Install Python dependencies

```powershell
python -m pip install -r requirements.txt
```

### Step 4 — Install Ollama and download the model

Install Ollama from:

[https://ollama.com/download](https://ollama.com/download)

Then run:

```powershell
ollama pull qwen2.5:3b
```

Make sure Ollama is running locally. The application expects its chat API at:

`http://127.0.0.1:11434/api/chat`

### Step 5 — Verify the trained model

Ensure these files exist in the `models/` directory:

- `prompt_classifier.joblib`
- `prompt_classifier_normalized.joblib`

The application expects the normalized model at:

`models/prompt_classifier_normalized.joblib`

### Step 6 — Start the API server

```powershell
python -m uvicorn api:app --reload
```

Keep the terminal open while using the application.

### Step 7 — Open the chatbot

- **Chat interface:** [http://127.0.0.1:8000/chat](http://127.0.0.1:8000/chat)
- **Interactive API documentation:** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **Health check:** [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Returns service status |
| GET | `/health` | Checks model and threshold loading |
| POST | `/analyze` | Analyzes a prompt without forwarding it |
| POST | `/gateway` | Analyzes a prompt and blocks or forwards it |
| GET | `/chat` | Serves the chatbot interface |

## Testing

Run the automated test suite from the project directory:

```powershell
python -m pytest -q
```

During development, the suite reported **62 passing tests**. Re-run the tests against the published repository to verify the current result.

Manual end-to-end checks also demonstrated:

- A suspicious prompt was rejected with HTTP `403`.
- A benign prompt was accepted with HTTP `200` and forwarded to the local LLM.

## Model Evaluation

The hybrid detector's development evaluation reported:

- **Accuracy:** 99.84%
- **Malicious-class precision:** 99.75%
- **Malicious-class recall:** 100%
- **False-positive rate:** 0.48%

These are development results, not an independent security benchmark. The evaluation data was inspected during development, so the results should not be treated as an unbiased estimate of performance on unseen attacks.

The hybrid results differ from the ML-only results. Refer to the evaluation scripts and reports for the methodology and exact experiment behind each metric.

## Evaluation Reports

The [`reports/`](https://github.com/Anushapeddapalli/AI-prompt-firewall/tree/main/reports) directory contains available model evaluation and analysis reports.

## Limitations

- Prompt-injection detection can produce false positives and false negatives.
- The classifier does not identify every possible malicious intent.
- The application analyzes submitted prompts and does not guarantee full conversation-level security.
- Requests sent directly to Ollama can bypass the gateway unless direct access is restricted.
- Performance depends on the model, threshold, detection rules, dataset, and distribution of prompts.

## Dataset and Attribution

This project uses the **Prompt Injection Detection Dataset** by Mirza Akhi.

- **Original source:** [Prompt Injection Detection Dataset](https://github.com/mirzaakhi/prompt-injection-detection-dataset)
- **Dataset citation:** Mirza Akhi. (2026). *Prompt Injection Detection Dataset (Version 1.0).* Zenodo.
- **DOI:** [https://doi.org/10.5281/zenodo.20817277](https://doi.org/10.5281/zenodo.20817277)
- **Dataset license:** Creative Commons Attribution 4.0 International (CC BY 4.0).

When redistributing the dataset, retain appropriate attribution, link to the license, and indicate whether modifications were made. Verify the license and provenance of any additional files in `data/`, including `prompts.csv`, before publishing them.

