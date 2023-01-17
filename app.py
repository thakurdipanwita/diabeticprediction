# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 15:17:25 2023

@author: Dipanwita
"""

import numpy as np
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)
# read our pickle file and label our logisticmodel as model
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST','GET'])
def predict():

    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    
    if prediction==0:
        return render_template('index.html',
                               prediction_text='Low chances of patient having diabetes'.format(prediction),
                               )
    else:
        return render_template('index.html',
                               prediction_text='High chances of patient having diabetes'.format(prediction),
                              )



if __name__ == "__main__":
    app.run(debug=True)