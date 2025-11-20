import os
import json
from flask import Flask, request, jsonify, send_from_directory
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import spacy
import numpy as np
import requests
from dotenv import load_dotenv

# Load .env
load_dotenv()

# Config
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
OLLAMA_CHAT_ENDPOINT = OLLAMA_HOST.rstrip("/") + "/api/chat"
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", 0.60))
TOP_K_CONTEXT = int(os.getenv("TOP_K_CONTEXT", 3))
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "phi3")
FAQ_PATH = os.getenv("FAQ_PATH", "faqs.json")

# Flask app
app = Flask(__name__)

# Load spaCy
nlp = spacy.load("en_core_web_sm")

# Load FAQs file
with open(FAQ_PATH, "r", encoding="utf-8") as f:
    FAQS = json.load(f)

faq_questions = [f["question"] for f in FAQS]
faq_answers = [f["answer"] for f in FAQS]

# -------- TEXT PREPROCESSING -------- #
def preprocess(text: str) -> str:
    doc = nlp(text.lower())
    tokens = [t.lemma_.strip() for t in doc if t.is_alpha and not t.is_stop]
    return " ".join(tokens)

# Preprocess FAQ questions
preprocessed_questions = [preprocess(q) for q in faq_questions]
vectorizer = TfidfVectorizer().fit(preprocessed_questions)
faq_tfidf = vectorizer.transform(preprocessed_questions)

# -------- FIND BEST FAQ -------- #
def find_best_faq(query: str):
    pre = preprocess(query)
    q_vec = vectorizer.transform([pre])
    sims = cosine_similarity(q_vec, faq_tfidf)[0]
    best_idx = int(np.argmax(sims))
    best_score = float(sims[best_idx])
    return best_idx, best_score, sims

# -------- CALL OLLAMA CHAT API -------- #
def call_ollama_chat(system_message: str, user_message: str, model: str = OLLAMA_MODEL, timeout: int = 120):
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ]
    }
    try:
        resp = requests.post(OLLAMA_CHAT_ENDPOINT, json=payload, timeout=timeout)
        resp.raise_for_status()
        text = resp.text.strip()

        # Attempt to extract first JSON object from the response
        start_idx = text.find("{")
        end_idx = text.rfind("}") + 1
        json_text = text[start_idx:end_idx]
        data = json.loads(json_text)

        # Check common keys
        if "message" in data and "content" in data["message"]:
            return data["message"]["content"]

        if "choices" in data and len(data["choices"]) > 0:
            m = data["choices"][0].get("message") or data["choices"][0]
            if isinstance(m, dict) and "content" in m:
                return m["content"]

        # Fallback: just stringify the JSON
        return str(data)

    except json.JSONDecodeError:
        return f"[Ollama JSON Error] {text}"
    except Exception as e:
        return f"[Ollama Error] {str(e)}"

# -------- MAIN API ENDPOINT -------- #
@app.route("/ask", methods=["POST"])
def ask():
    body = request.get_json(force=True)
    question = body.get("question") or body.get("message") or body.get("text") or ""

    if not question:
        return jsonify({"error": "no question provided"}), 400

    # FAQ similarity
    best_idx, best_score, sims = find_best_faq(question)
    best_faq = FAQS[best_idx]

    if best_score >= SIMILARITY_THRESHOLD:
        return jsonify({
            "source": "faq",
            "answer": best_faq["answer"],
            "faq_id": best_faq.get("id"),
            "score": round(best_score, 4)
        })

    # Fallback to Ollama
    sim_arr = sims
    topk = min(TOP_K_CONTEXT, len(FAQS))
    topk_idxs = list(np.argsort(sim_arr)[-topk:][::-1])
    context_text = "\n\n".join([f"Q: {FAQS[i]['question']}\nA: {FAQS[i]['answer']}" for i in topk_idxs])

    system_prompt = (
        "You are a helpful support assistant using provided FAQ context. "
        "If the answer is not found in context, answer naturally and concisely."
    )
    combined_user = f"User question: {question}\n\nRelevant FAQ context:\n{context_text}"

    generated_reply = call_ollama_chat(
        system_message=system_prompt,
        user_message=combined_user,
        model=os.getenv("OLLAMA_MODEL", "phi3"),
        timeout=60
    )

    return jsonify({
        "source": "ollama",
        "answer": generated_reply,
        "score": round(best_score, 4),
        "faq_candidates": [
            {
                "id": FAQS[i].get("id"),
                "question": FAQS[i]["question"],
                "score": round(float(sim_arr[i]), 4)
            } for i in topk_idxs
        ]
    })

# -------- SERVE CHAT UI -------- #
@app.route("/", methods=["GET"])
def home():
    # Serve index.html from same folder
    return send_from_directory('.', 'index.html')

# -------- RUN APP -------- #
if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.getenv("PORT", 5000)), debug=True)
