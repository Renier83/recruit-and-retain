# Load libraries
import flask
import pandas as pd
import numpy as np
from flask import Flask, request, render_template, jsonify
import tensorflow as tf
import keras
from keras.models import load_model
from keras.models import Sequential


# instantiate flask 
app = flask.Flask(__name__)

#def load_model_to_app():
predmodel = load_model('neural_model.h5')


@app.route('/')
def home():
    return render_template('index.html')

# define a predict function as an endpoint 
@app.route("/predict", methods=["POST"])
def predict():

    '''
    For rendering results on HTML GUI
    '''
    
    # retrieving values from form
    init_features = [float(x) for x in request.form.values()]
    final_features = np.array([np.asarray(init_features)])

    prediction = predmodel.predict(final_features) # making prediction

    #class_ = np.where(prediction == np.amax(prediction, axis=1))[1][0]

    return render_template('index.html', prediction_text='Attrition Likelihood: {}'.format(prediction)) # rendering the predicted result

if __name__ == "__main__":
    app.run(debug=True)



#     data = {"success": False}

#     params = flask.request.json
#     if (params == None):
#         params = flask.request.args

#     # if parameters are found, return a prediction
#     if (params != None):
#         x=pd.DataFrame.from_dict(params, orient='index').transpose()
#         with graph.as_default():
#             data["prediction"] = str(model.predict(x)[0][0])
#             data["success"] = True

#     # return a response in json format 
#     return flask.jsonify(data)    

# # start the flask app, allow remote connections 
# app.run(host='0.0.0.0')