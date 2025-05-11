from flask import Flask, render_template, request, redirect, session
import os
import joblib
import pandas as pd
import re
import numpy as np
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
from datetime import datetime

app = Flask(__name__)
app.secret_key = os.urandom(24)

# === Paths ===
MODEL_PATH = 'model/passmodel3.pkl'
TOKENIZER_PATH = 'model/tfidfvectorizer3.pkl'
DATA_PATH = 'data/drugsComTrain_raw.csv'
LOG_PATH = 'data/tested_cases.csv'  # <-- path to store tested inputs

# === Load model and vectorizer ===
model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(TOKENIZER_PATH)

# === NLP Setup ===
stop = stopwords.words('english')
lemmatizer = WordNetLemmatizer()

# === Routes ===

@app.route('/')
def login():
    return render_template('login.html')

@app.route("/logout")
def logout():
    session.clear()
    return redirect('/')

@app.route('/index')
def index():
    if 'user_id' in session:
        return render_template('home.html')
    else:
        return redirect('/')

@app.route('/login.validation', methods=['POST'])
def login_validation():
    username = request.form.get('username')
    password = request.form.get('password')

    session['user_id'] = username

    if username == "priyansh@gmail.com" and password == "priyansh":
        return render_template('home.html')
    else:
        err = "Priyanshu caught you"
        return render_template('login.html', lbl=err)

@app.route('/predict', methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        raw_text = request.form.get('rawtext', '')

        if raw_text.strip() == "":
            return render_template('predict.html', rawtext="No input provided", result="None", top_drugs=[])

        clean_text = cleanText(raw_text)
        clean_lst = [clean_text]

        # Vectorize and predict
        tfidf_vect = vectorizer.transform(clean_lst)
        prediction = model.predict(tfidf_vect)
        predicted_cond = prediction[0]

        # Load drug data and extract top drugs
        df = pd.read_csv(DATA_PATH)
        top_drugs = top_drugs_extractor(predicted_cond, df)

        # Save tested input to CSV
        save_tested_case(raw_text, predicted_cond)

        return render_template('predict.html', rawtext=raw_text, result=predicted_cond, top_drugs=top_drugs)

    return redirect('/index')


@app.route('/view_tests')
def view_tests():
    if not os.path.exists(LOG_PATH):
        tested_cases = []
    else:
        df_log = pd.read_csv(LOG_PATH)
        tested_cases = df_log.to_dict(orient='records')
    return render_template('view_tests.html', tested_cases=tested_cases)

@app.route('/home')
def homepage():
    return render_template('home.html')



# === Helpers ===

def cleanText(raw_review):
    review_text = BeautifulSoup(raw_review, 'html.parser').get_text()
    letters_only = re.sub('[^a-zA-Z]', ' ', review_text)
    words = letters_only.lower().split()
    meaningful_words = [w for w in words if w not in stop]
    lemmatized_words = [lemmatizer.lemmatize(w) for w in meaningful_words]
    return ' '.join(lemmatized_words)

def top_drugs_extractor(condition, df):
    df_top = df[(df['rating'] >= 9) & (df['usefulCount'] >= 100)].sort_values(by=['rating', 'usefulCount'], ascending=[False, False])
    drug_lst = df_top[df_top['condition'] == condition]['drugName'].head(5).tolist()
    return drug_lst

def save_tested_case(sentence, condition):
    log_entry = pd.DataFrame([{
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'input': sentence,
        'predicted_condition': condition
    }])
    
    if os.path.exists(LOG_PATH):
        existing = pd.read_csv(LOG_PATH)
        combined = pd.concat([existing, log_entry], ignore_index=True)
    else:
        combined = log_entry
    
    combined.to_csv(LOG_PATH, index=False)

# === Run Server ===

if __name__ == "__main__":
    app.run(debug=True, host="localhost", port=8080)
