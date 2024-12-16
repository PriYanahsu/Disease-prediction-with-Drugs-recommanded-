from flask import Flask, render_template,request,redirect,session
import os
import joblib
import pandas as pd
import re
import numpy as np
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from bs4 import BeautifulSoup

HTML_WRAPPER = """<div style="overflow-x: auto; border: 1px solid #e6e9ef; border-radius:0.25rem; padding: 1rem">{}</div>"""

app = Flask(__name__)

app.secret_key=os.urandom(24)

Model_Path = 'model/passmodel2.pkl'

TOKENIZER_PATH = 'model/tfidfvectorizer2.pkl'

DATA_PATH = 'data/drugsComTrain_raw.csv'

vectorizer = joblib.load(TOKENIZER_PATH)

model = joblib.load(Model_Path)

stop = stopwords.words('english')
lemmatizer = WordNetLemmatizer()

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

    session['user_id']= username

    if username=="admin@gmail.com" and password =="admin":
        return render_template('home.html')
    
    else:
        err="Priyanshu caught you"
        return render_template('login.html', lbl=err)
    
    return ""

@app.route('/predict', methods=["GET","POST"])
def predict():
    if request.method == "POST":
        raw_text = request.form['rawtext']

        if raw_text != "":
            clean_text = cleanText(raw_text)
            clean_lst = [clean_text]

            tfidf_vect = vectorizer.transform(clean_lst)
            prediction = model.predict(tfidf_vect)
            predicted_cond = prediction[0]
            df = pd.read_csv(DATA_PATH)
            top_drugs = top_drugs_extractor(predicted_cond,df)

            return render_template('predict.html', rawtext= raw_text, result = predicted_cond, top_drugs = top_drugs)
        
        else:
            raw_text = "There is no text selected"

def cleanText(raw_review):
    # delte HTML
    review_text = BeautifulSoup(raw_review, 'html.parser').get_text()
    # Make space
    letters_only = re.sub('[^a-zA-Z]', ' ', review_text)
    # lower letter
    words = letters_only.lower().split()
    # Stopwords
    meaningful_words = [w for w in words if not w in stop]
    # lemmitizer
    lemmitize_words = [lemmatizer.lemmatize(w) for w in meaningful_words]
    # space join words
    return(' '.join(lemmitize_words))

def top_drugs_extractor(condition,df):
    # Filter the DataFrame based on rating and usefulCount
    df_top = df[(df['rating'] >= 9) & (df['usefulCount'] >= 100)].sort_values(by=['rating', 'usefulCount'], ascending=[False, False])
    
    # Extract the top 5 drugs for the specified condition
    drug_lst = df_top[df_top['condition'] == condition]['drugName'].head(5).tolist()
    return drug_lst

from app import app
if __name__ == "__main__":

    app.run(debug=True, host="localhost", port=8080)

# from waitress import serve
# from myapp import app  # Replace with the actual name of your Flask app

# if __name__ == "__main__":
#     server(app, host="0.0.0.0", port=5000)


# python -m flask run
