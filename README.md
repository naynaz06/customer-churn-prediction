# Smart Customer Support Ticket Intelligence System

## Project Overview
This project uses Natural Language Processing (NLP) and Machine Learning to automatically classify customer support tickets into predefined categories. It helps organize customer queries and can support faster ticket routing.

## Features
- Customer ticket text preprocessing
- Bag of Words (BoW) and TF-IDF feature extraction
- Comparison of Naive Bayes, Logistic Regression, and Linear SVM models
- Automated ticket category prediction
- Interactive Streamlit web application

## Technologies Used
- Python
- Pandas
- Scikit-learn
- NLTK
- TF-IDF
- Linear Support Vector Classifier (LinearSVC)
- Streamlit
- Joblib
- Jupyter Notebook



## Dataset
The project uses the Banking77 customer support dataset, containing 77 categories of banking-related customer queries.

- Training samples: 10,003
- Test samples: 3,080
- Number of categories: 77

## Model
LinearSVC with TF-IDF feature extraction was selected as the final model.

The model and TF-IDF vectorizer are saved in the `models/` directory for use in the Streamlit application.

## Run the Application

1. Clone or download this repository.
2. Install the required dependencies.
3. Open a terminal in the project directory.
4.pip install pandas scikit-learn nltk streamlit joblib
5. Run(in Terminal):


python -m streamlit run app.py

## Application Preview
![Smart Customer Support Ticket Classifier](Smart%20Customer%20Support%20Ticket%20Classifier.png)


## Future Improvements
- Improve classification accuracy for confusing ticket categories.
- Explore additional NLP and machine learning techniques.
- Expand the application with more customer support features.
