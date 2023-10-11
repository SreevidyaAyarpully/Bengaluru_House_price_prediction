from flask import Flask, render_template, request
import pickle
import numpy as np

model = pickle.load(open('model.pkl','rb'))
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_house_price():
    area_type = float(request.form.get('area_type'))
    bedrooms = int(request.form.get('bedrooms'))
    total_sqft = float(request.form.get('total_sqft'))
    bathroom = float(request.form.get('bathroom'))
    balcony = float(request.form.get('balcony'))

    #prediction
    result = model.predict(np.array([area_type, bedrooms, total_sqft, bathroom,balcony ]).reshape(1,5))

    result = f"Price : {result[0]:.2f} lakhs"

    return render_template('index.html',result=result)


if __name__ == '__main__':
    app.run(host='0.0.0.0',port=8080)