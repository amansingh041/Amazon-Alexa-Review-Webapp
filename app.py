from flask import Flask, render_template, request, redirect, url_for
import pickle
import string
from nltk.corpus import stopwords
import nltk
nltk.download('punkt_tab')
nltk.download('stopwords')
from nltk.stem import WordNetLemmatizer
wl = WordNetLemmatizer()
nltk.download('wordnet')

# Create Flask app
app = Flask(__name__)

# Load trained model and vectorizer
model = pickle.load(open('alexa_review_model_rfc.pkl', 'rb'))
vectorizer = pickle.load(open('alexa_review_vectorizer_tfidf.pkl', 'rb'))



def transformer_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    L = []
    for i in text:
        if i.isalnum():
            L.append(i)
    
    text = L.copy()
    L.clear()
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            L.append(i)
    text = L.copy()
    L.clear()
    for i in text:
        L.append(wl.lemmatize(i,"v"))

    return " ".join(L)



# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    review = request.form.get('review')
    transformed_review = transformer_text(review)

    # Transform input using vectorizer
    data = vectorizer.transform([transformed_review])

    # Predict
    prediction = model.predict(data)

    # Output
    if prediction[0] == 1:
        result = "Positive 😊"
    else:
        result = "Negative 😡"

    return redirect(url_for('result', output=result))

# Result route (GET)
@app.route('/result')
def result():
    output = request.args.get('output')

    return render_template('index.html', prediction_text=output)


if __name__ == "__main__":
    app.run(debug=True)