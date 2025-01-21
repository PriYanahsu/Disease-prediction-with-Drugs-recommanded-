# Disease Prediction with Drug Recommendation - README

## Overview
The Disease Prediction with Drug Recommendation is a machine learning project designed to predict diseases based on symptoms and recommend suitable drugs. This project uses Jupyter Notebook for model training and evaluation, while Flask is used to build a web-based interface for real-time predictions and recommendations.

## Features
#### * Model Training: Train a machine learning model in Jupyter Notebook to predict diseases and recommend drugs.
#### * Interactive Interface: Flask provides a lightweight and user-friendly interface for making predictions.
#### * Data Visualization: Includes visual insights such as word clouds to represent symptom correlations.
#### * Simple Deployment: The Flask app runs locally with a single command.

## Technologies Used
#### 1.Python: Core programming language.
#### 2.Jupyter Notebook:  For exploratory data analysis (EDA), model training, and evaluation.
#### 3.Flask: For building the backend and hosting the web interface.

## Libraries:
#### 1.Pandas: Data manipulation and analysis.
#### 2.Scikit-learn: Model training and evaluation.
#### 3.NLTK: For text preprocessing (if applicable).
#### 4.Flask: Web app framework.
#### 5.WordCloud: To visualize symptom data relationships.

## Project Structure

Spam-Message-Detector

  #### notebooks
     1.model_training.ipynb    # Jupyter notebook for model training and evaluation
      static
       2. assets               # Static files such as CSS and images
     templates
       3.index.html            # HTML template for the Flask frontend
      4.app.py                    # Flask app script
      5.model.pkl                 # Trained model file (generated after training)
      6.requirements.txt          # List of Python dependencies
      7.README.md                 # Project documentation (this file)

## How to Run the Project
### 1. Clone the Repository
Clone the repository to your local machine:

    git clone <repository-url>
    cd Disease-Prediction-With-Drugs-Recommendation

### 2. Install Dependencies
Install the required Python libraries:

    pip install -r requirements.txt

### 3. Train the Model
If you'd like to retrain the model:

1. Open notebooks/model_training.ipynb in Jupyter Notebook.
2. Follow the steps in the notebook to:
   * Load the dataset.
   * Preprocess the text data.
   * Train and evaluate the machine learning model.
3. Save the trained model as model.pkl.

#### Note: A pre-trained model is already included in the repository.

### 4. Run the Flask
Run the flask app with the following command:

    python -m flask run

### 5. Open the App in Your Browser
After running the above command, Streamlit will start a local server and provide a URL. Open the URL in your browser (default: http://localhost:8501).

## Screenshot
 ### 1. Login first
![Screenshot (248)](https://github.com/user-attachments/assets/559433ae-75f7-4510-a850-d874f453ec86)

 ### 2. This section where you and put your parameter how you feel and whats the problem occur with your body in daily routine.
![Screenshot (251)](https://github.com/user-attachments/assets/6eed16ec-04c4-41fb-b9d8-5ef5b0055d8f)

 ### 3. After that you add paramtere and press the predict
 ![Screenshot (249)](https://github.com/user-attachments/assets/25d8429a-bdb5-4e80-b1fb-951a61e67b79)

 ### 4. Final output will be Disease and best five recommanded drugs
![Screenshot (250)](https://github.com/user-attachments/assets/78522e3d-f5ad-4dcf-944d-9fff22d2ec96)


## Additional Features
  #### Disease Prediction: Input symptoms to predict the most likely disease.
  #### Drug Recommendation: Get five drug suggestions for the predicted disease.
  #### Word Cloud Visualization: The app includes a word cloud to visualize the most common symptoms and their associations and priorities the words that are in specified disease.


