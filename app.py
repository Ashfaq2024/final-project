from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

model = joblib.load('model/bug_classifier.pkl')
vectorizer = joblib.load('model/vectorizer.pkl')

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = ''
    if request.method == 'POST':
        bug_desc = request.form['bug']
        vec = vectorizer.transform([bug_desc])
        prediction = model.predict(vec)[0]
    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
