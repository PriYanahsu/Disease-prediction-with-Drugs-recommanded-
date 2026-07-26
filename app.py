from flask import Flask, render_template, request, redirect, session, url_for
import os
import csv
import joblib
import numpy as np
import pandas as pd
import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from datetime import datetime
from collections import Counter, defaultdict
from functools import wraps
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "dev-only-change-me")

LOGIN_USER = os.environ.get("LOGIN_USER", "priyansh@gmail.com")
LOGIN_PASS = os.environ.get("LOGIN_PASS", "priyansh")

MODEL_PATH = "model/passmodelAce.pkl"
TOKENIZER_PATH = "model/tfidfvectorizerAce.pkl"
DATA_PATH = "data/drugsComTrain_raw.csv"
LOG_PATH = "data/tested_cases.csv"
DRUG_CACHE_PATH = "data/top_drugs_cache.csv"

model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(TOKENIZER_PATH)

# frozenset = O(1) membership; WordNetLemmatizer is reused across requests
STOP_WORDS = frozenset(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()
_TAG_RE = re.compile(r"<[^>]+>")
_NON_ALPHA_RE = re.compile(r"[^a-zA-Z]+")

# Warm WordNet + sklearn so the first real request is not a 2s cold start
_ = lemmatizer.lemmatize("symptoms")
_ = model.predict(vectorizer.transform(["fever cough headache"]))

MODEL_META = {
    "name": "PassiveAggressiveClassifier",
    "vectorizer": "TF-IDF",
    "classes": int(len(getattr(model, "classes_", []))),
    "task": "Multi-class symptom -> condition",
}


def build_top_drugs_map(csv_path: str) -> dict:
    """Precompute condition -> top-5 drugs once (avoids scanning 79MB CSV per request)."""
    if os.path.exists(DRUG_CACHE_PATH):
        cache = pd.read_csv(DRUG_CACHE_PATH)
        mapping = defaultdict(list)
        for _, row in cache.iterrows():
            mapping[str(row["condition"])].append(str(row["drugName"]))
        return dict(mapping)

    df = pd.read_csv(
        csv_path,
        usecols=["drugName", "condition", "rating", "usefulCount"],
        low_memory=False,
    )
    df = df.dropna(subset=["condition", "drugName"])
    df = df[(df["rating"] >= 9) & (df["usefulCount"] >= 100)]
    df = df.sort_values(["condition", "rating", "usefulCount"], ascending=[True, False, False])

    rows = []
    mapping = {}
    for condition, group in df.groupby("condition", sort=False):
        # unique drug names preserving rating order
        drugs = list(dict.fromkeys(group["drugName"].tolist()))[:5]
        mapping[str(condition)] = drugs
        for drug in drugs:
            rows.append({"condition": condition, "drugName": drug})

    pd.DataFrame(rows).to_csv(DRUG_CACHE_PATH, index=False)
    return mapping


TOP_DRUGS_BY_CONDITION = build_top_drugs_map(DATA_PATH)


def login_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if "user_id" not in session:
            return redirect(url_for("login"))
        return view(*args, **kwargs)

    return wrapped


@app.context_processor
def inject_globals():
    return {"show_nav": False, "model_meta": MODEL_META}


@app.route("/")
def login():
    if "user_id" in session:
        return redirect(url_for("index"))
    return render_template("login.html", show_nav=False)


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


@app.route("/index")
@login_required
def index():
    return render_template("home.html", show_nav=True)


@app.route("/home")
@login_required
def homepage():
    return redirect(url_for("index"))


@app.route("/login.validation", methods=["POST"])
def login_validation():
    username = (request.form.get("username") or "").strip()
    password = request.form.get("password") or ""

    if username == LOGIN_USER and password == LOGIN_PASS:
        session["user_id"] = username
        return redirect(url_for("index"))

    return render_template(
        "login.html",
        show_nav=False,
        error="Invalid email or password. Please try again.",
    )


@app.route("/predict", methods=["GET", "POST"])
@login_required
def predict():
    if request.method != "POST":
        return redirect(url_for("index"))

    name = (request.form.get("name") or "").strip()
    age = (request.form.get("age") or "").strip()
    gender = (request.form.get("gender") or "").strip()
    height = (request.form.get("height") or "").strip()
    weight = (request.form.get("weight") or "").strip()
    location = (request.form.get("location") or "").strip()
    raw_text = (request.form.get("rawtext") or "").strip()

    if not raw_text:
        return render_template(
            "predict.html",
            show_nav=True,
            name=name,
            age=age,
            gender=gender,
            height=height,
            weight=weight,
            location=location,
            rawtext="",
            result=None,
            confidence=None,
            top_predictions=[],
            top_drugs=[],
            tokens=[],
            error="Please describe your symptoms before predicting.",
        )

    clean_text = clean_text_fast(raw_text)
    tokens = [t for t in clean_text.split() if t][:18]
    tfidf_vect = vectorizer.transform([clean_text])
    predicted_cond, confidence, top_predictions = rank_predictions(tfidf_vect, top_k=3)
    top_drugs = TOP_DRUGS_BY_CONDITION.get(str(predicted_cond), [])
    save_tested_case(name, age, gender, height, weight, location, raw_text, predicted_cond)

    return render_template(
        "predict.html",
        show_nav=True,
        name=name,
        age=age,
        gender=gender,
        height=height,
        weight=weight,
        location=location,
        rawtext=raw_text,
        result=predicted_cond,
        confidence=confidence,
        top_predictions=top_predictions,
        top_drugs=top_drugs,
        tokens=tokens,
        error=None,
    )


@app.route("/view_tests")
@login_required
def view_tests():
    tested_cases = load_tested_cases()
    return render_template("view_tests.html", show_nav=True, tested_cases=tested_cases)


@app.route("/clear_history", methods=["POST"])
@login_required
def clear_history():
    try:
        if os.path.exists(LOG_PATH):
            os.remove(LOG_PATH)
    except Exception as e:
        print(f"Error clearing history: {e}")
    return redirect(url_for("view_tests"))


@app.route("/analytics")
@login_required
def analytics():
    try:
        tested_cases = load_tested_cases()
        if not tested_cases:
            return render_template(
                "analytics.html",
                show_nav=True,
                condition_counts={},
                drug_counts={},
                daily_counts={},
                total_predictions=0,
            )

        conditions = [c.get("predicted_condition") for c in tested_cases if c.get("predicted_condition")]
        condition_counts = dict(Counter(conditions))

        timestamps = []
        for case in tested_cases:
            ts = case.get("timestamp")
            if ts:
                timestamps.append(str(ts)[:10])
        daily_counts = dict(sorted(Counter(timestamps).items()))

        return render_template(
            "analytics.html",
            show_nav=True,
            condition_counts=condition_counts,
            drug_counts={},
            daily_counts=daily_counts,
            total_predictions=len(conditions),
        )
    except Exception as e:
        print(f"Error processing analytics: {e}")
        return render_template(
            "analytics.html",
            show_nav=True,
            condition_counts={},
            drug_counts={},
            daily_counts={},
            total_predictions=0,
        )


def clean_text_fast(raw_review: str) -> str:
    """Lightweight cleaner — no BeautifulSoup round-trip on plain symptom text."""
    text = _TAG_RE.sub(" ", raw_review)
    text = _NON_ALPHA_RE.sub(" ", text).lower()
    words = [w for w in text.split() if w not in STOP_WORDS and len(w) > 1]
    return " ".join(lemmatizer.lemmatize(w) for w in words)


def rank_predictions(tfidf_vect, top_k=3):
    """Softmax over decision scores → confidence + alternate class rankings."""
    scores = np.asarray(model.decision_function(tfidf_vect)).reshape(-1)
    classes = np.asarray(model.classes_)
    # numerically stable softmax
    shifted = scores - scores.max()
    probs = np.exp(shifted)
    probs = probs / probs.sum()

    order = np.argsort(probs)[::-1][:top_k]
    ranked = [
        {
            "label": str(classes[i]),
            "confidence": round(float(probs[i]) * 100, 1),
            "score": round(float(scores[i]), 3),
        }
        for i in order
    ]
    return ranked[0]["label"], ranked[0]["confidence"], ranked


def save_tested_case(name, age, gender, height, weight, location, rawtext, condition):
    """Append a single row — do not rewrite the whole CSV every prediction."""
    fieldnames = [
        "timestamp",
        "name",
        "age",
        "gender",
        "height",
        "weight",
        "location",
        "input",
        "predicted_condition",
    ]
    row = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "name": name,
        "age": age,
        "gender": gender,
        "height": height,
        "weight": weight,
        "location": location,
        "input": rawtext,
        "predicted_condition": condition,
    }

    os.makedirs(os.path.dirname(LOG_PATH) or ".", exist_ok=True)
    write_header = not os.path.exists(LOG_PATH) or os.path.getsize(LOG_PATH) == 0
    try:
        with open(LOG_PATH, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerow(row)
    except Exception as e:
        print(f"Error saving to CSV: {e}")


def load_tested_cases():
    if not os.path.exists(LOG_PATH) or os.path.getsize(LOG_PATH) == 0:
        return []
    try:
        with open(LOG_PATH, newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))
    except Exception as e:
        print(f"Error reading log: {e}")
        return []


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(debug=os.environ.get("FLASK_DEBUG") == "1", host="0.0.0.0", port=port)
