import json
import pandas as pd
import pymysql
from sqlalchemy import create_engine
import sqlalchemy
from flask import Flask, request, render_template, jsonify
import os
import pickle
import numpy as np


model = pickle.load(open('model.pkl', 'rb')) # loading the trained model


# Initialize Flask application
app = Flask(__name__)

# Set up your default route


@app.route('/')
def home():
    return render_template('machinelearning.html')


@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    
    # retrieving values from form
    init_features = [float(x) for x in request.form.values()]
    final_features = [np.array(init_features)]

    prediction = model.predict(final_features) # making prediction


    return render_template('machinelearning.html', prediction_text='Attrition Likelihood: {}'.format(prediction)) # rendering the predicted result




if __name__ == "__main__":
    app.run(debug=True)
