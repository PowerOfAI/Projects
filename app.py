

import numpy as np
# pip install flask
from flask import Flask, render_template, request, jsonify
import pickle


app = Flask(__name__)

model = pickle.load(open("model.pkl", "rb"))


@app.route("/")    
def home():
    return render_template('index.html')

@app.route("/predict", methods=["POST"])
def predict():
    
    #from sklearn.preprocessing import StandardScaler
    #scaler = StandardScaler()
    features = [int(x) for x in request.form.values()]  
    final_features = [np.array(features)]
    #final_features_norm = scaler.fit_transform(final_features)
    prediction = model.predict(final_features)   
    output = round(prediction[0])
    return render_template("index.html", prediction_text = "The rating of the retailer is {} out of 10.".format(output),
                           rating_ref = 'Rating Ref: Higher the rating, more credit worthy the retailer is.')

@app.route("/predict_customer_rating", methods=["POST"])
def predict_customer_rating():
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])
    
    output = round(prediction[0],2)
    return jsonify(output)
    

if __name__ == "__main__":
    app.run(debug=True)




