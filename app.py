from flask import Flask, render_template, request, redirect, session, url_for
import os
import joblib
import pandas as pd
import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
from datetime import datetime
from collections import Counter
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

model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(TOKENIZER_PATH)
df_drugs = pd.read_csv(DATA_PATH)

stop = stopwords.words("english")
lemmatizer = WordNetLemmatizer()


def login_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if "user_id" not in session:
            return redirect(url_for("login"))
        return view(*args, **kwargs)

    return wrapped


@app.route("/")
def login():
    if "user_id" in session:
        return redirect(url_for("index"))
    return render_template("login.html")


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


@app.route("/index")
@login_required
def index():
    return render_template("home.html")


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
        error="Invalid email or password. Please try again.",
    )


@app.route("/predict", methods=["GET", "POST"])
@login_required
def predict():
    if request.method != "POST":
        return redirect(url_for("index"))

    name = request.form.get("name", "")
    age = request.form.get("age", "")
    gender = request.form.get("gender", "")
    height = request.form.get("height", "")
    weight = request.form.get("weight", "")
    location = request.form.get("location", "")
    raw_text = request.form.get("rawtext", "")

    if not raw_text.strip():
        return render_template(
            "predict.html",
            name=name,
            age=age,
            gender=gender,
            height=height,
            weight=weight,
            location=location,
            rawtext="",
            result=None,
            top_drugs=[],
            error="Please describe your symptoms before predicting.",
        )

    clean_text = cleanText(raw_text)
    tfidf_vect = vectorizer.transform([clean_text])
    prediction = model.predict(tfidf_vect)
    predicted_cond = prediction[0]
    top_drugs = top_drugs_extractor(predicted_cond, df_drugs)
    save_tested_case(name, age, gender, height, weight, location, raw_text, predicted_cond)

    return render_template(
        "predict.html",
        name=name,
        age=age,
        gender=gender,
        height=height,
        weight=weight,
        location=location,
        rawtext=raw_text,
        result=predicted_cond,
        top_drugs=top_drugs,
        error=None,
    )


@app.route("/view_tests")
@login_required
def view_tests():
    if not os.path.exists(LOG_PATH):
        tested_cases = []
    else:
        df_log = pd.read_csv(LOG_PATH)
        tested_cases = df_log.fillna("").to_dict(orient="records")
    return render_template("view_tests.html", tested_cases=tested_cases)


@app.route("/analytics")
@login_required
def analytics():
    try:
        if os.path.exists(LOG_PATH):
            df = pd.read_csv(LOG_PATH)
        else:
            df = pd.DataFrame()

        condition_counts = (
            dict(Counter(df["predicted_condition"].dropna().tolist())) if not df.empty else {}
        )

        drugs = []
        if "top_drugs" in df.columns and not df.empty:
            for drug_list in df["top_drugs"]:
                drugs.extend([d.strip() for d in str(drug_list).split(",") if d.strip()])
        drug_counts = dict(Counter(drugs)) if drugs else {}

        timestamps = pd.to_datetime(df["timestamp"], errors="coerce") if not df.empty else pd.Series(dtype="datetime64[ns]")
        daily_counts = timestamps.dt.date.value_counts().sort_index() if not df.empty else {}
        daily_counts_dict = {str(key): int(value) for key, value in daily_counts.items()}

        total_predictions = int(sum(condition_counts.values())) if condition_counts else 0

        return render_template(
            "analytics.html",
            condition_counts=condition_counts,
            drug_counts=drug_counts,
            daily_counts=daily_counts_dict,
            total_predictions=total_predictions,
        )
    except Exception as e:
        print(f"Error processing analytics: {e}")
        return render_template(
            "analytics.html",
            condition_counts={},
            drug_counts={},
            daily_counts={},
            total_predictions=0,
        )


def cleanText(raw_review):
    review_text = BeautifulSoup(raw_review, "html.parser").get_text()
    letters_only = re.sub("[^a-zA-Z]", " ", review_text)
    words = letters_only.lower().split()
    meaningful_words = [w for w in words if w not in stop]
    lemmatized_words = [lemmatizer.lemmatize(w) for w in meaningful_words]
    return " ".join(lemmatized_words)


def top_drugs_extractor(condition, df):
    df_top = df[(df["rating"] >= 9) & (df["usefulCount"] >= 100)].sort_values(
        by=["rating", "usefulCount"], ascending=[False, False]
    )
    return df_top[df_top["condition"] == condition]["drugName"].head(5).tolist()


def save_tested_case(name, age, gender, height, weight, location, rawtext, condition):
    log_entry = pd.DataFrame(
        [
            {
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
        ]
    )

    if os.path.exists(LOG_PATH):
        existing = pd.read_csv(LOG_PATH)
        combined = pd.concat([existing, log_entry], ignore_index=True)
    else:
        combined = log_entry

    try:
        combined.to_csv(LOG_PATH, index=False)
    except Exception as e:
        print(f"Error saving to CSV: {e}")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(debug=os.environ.get("FLASK_DEBUG") == "1", host="0.0.0.0", port=port)
