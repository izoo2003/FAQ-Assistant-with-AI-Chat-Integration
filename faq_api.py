from flask import Flask, request, jsonify
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load your FAQ data (local file)
with open("faqs.json", "r") as f:
    faqs = json.load(f)

questions = [item["question"] for item in faqs]
answers = [item["answer"] for item in faqs]

# Convert all FAQ questions into vectors
vectorizer = TfidfVectorizer()
question_vectors = vectorizer.fit_transform(questions)

def find_best_answer(user_question):
    user_vec = vectorizer.transform([user_question])
    similarities = cosine_similarity(user_vec, question_vectors)
    best_index = np.argmax(similarities)
    return answers[best_index]

@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    user_question = data.get("question")
    answer = find_best_answer(user_question)
    return jsonify({"answer": answer})

if __name__ == "__main__":
    app.run(port=5000)
