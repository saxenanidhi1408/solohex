from flask import Flask,render_template,request
from model import predict_news

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    news = request.form['news']
    if not news.strip():
        return render_template('index.html',error="Enter text")

    result = predict_news(news)
    return render_template('result.html',news=news,prediction=result)

app.run(debug=False)