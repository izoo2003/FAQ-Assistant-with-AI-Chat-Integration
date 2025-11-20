# FAQ Assistant with AI Chat Integration

A local AI-powered FAQ chatbot that intelligently answers user queries. It matches questions against a pre-defined FAQ dataset using NLP and cosine similarity, and falls back to the Ollama `phi3` model for unmatched questions. Built with Python, Flask, and spaCy, it runs offline with a simple HTML chat interface.

---

## Features

- NLP-based FAQ matching for fast responses.
- Fallback to AI (Ollama phi3) when the question is not found in FAQs.
- Configurable similarity threshold and top-K context for AI.
- Simple web-based chat interface.
- Fully offline/local execution; no external webhook needed.

---

## Tech Stack

- Python 3.11+
- Flask
- spaCy (`en_core_web_sm`)
- scikit-learn
- NumPy
- Requests
- Ollama LLM (`phi3` model)
- HTML/JS for chat UI

---

## Installation

1. Clone the repository:

```bash
git clone <your-repo-url>
cd faq-ai-chatbot

pip install -r requirements.txt
python -m spacy download en_core_web_sm
