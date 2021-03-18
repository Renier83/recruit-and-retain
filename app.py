import json
import pandas as pd
import pymysql
from sqlalchemy import create_engine
import sqlalchemy
from flask import Flask, request, render_template, jsonify
import os
import pickle
import numpy as np



# Initialize Flask application
app = Flask(__name__)

# Use pickle to load in the pre-trained model.

model = pickle.load(open('model.pkl', 'rb')) # loading the trained model


# Set up your default route
@app.route('/')
def home():
    #print("/index")
    return render_template('index.html')


@app.route('/machinelearning')
def machinelearning():
    print("/machinelearning")
    return render_template('machinelearning.html')


@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    if request.method == 'POST':

        age = request.form['Age']
        distance = request.form['DistanceFromHome']
        level = request.form['JobLevel']
        satisfaction = request.form['JobSatisfaction']
        income = request.form['MonthlyIncome']
        hours = request.form['StandardHours']
        workyears = request.form['TotalWorkingYears']
        balance = request.form['WorkLifeBalance']
        role = request.form['YearsInCurrentRole']
        promotion = request.form['YearsSinceLastPromotion']
        mgr = request.form['YearsWithCurrManager']

        input_variables = pd.DataFrame([[age, distance, level, satisfaction,
                    income, hours, workyears,balance,role,promotion,mgr]],
                    columns=['age', 'distance', 'level', 'satisfaction',
                    'income', 'hours', 'workyears','balance','role','promotion','mgr'],
                    dtype=float)
        
        prediction = model.predict(input_variables)[0]
        
        return render_template('machinelearning.html', original_input = {
                'Age' : age,
                'DistanceFromHome': distance,
                'JobLevel': level,
                'JobSatisfaction': satisfaction,
                'MonthlyIncome': income,
                'StandardHours': hours,
                'TotalWorkingYears': workyears,
                'WorkLifeBalance': balance,
                'YearsInCurrentRole': role,
                'YearsSinceLastPromotion': promotion,
                'YearsWithCurrManager': mgr
            }, result=prediction,
        
        prediction_text='Attrition Likelihood: {}'.format(prediction),  #returns the values entered
        age_text='Age: {}'.format(age),
        distance_text='DistanceFromHome: {}'.format(distance), 
        level_text='JobLevel: {}'.format(level), 
        satisfaction_text='JobSatisfaction: {}'.format(satisfaction),
        income_text='MonthlyIncome: {}'.format(income), 
        hours_text='StandardHours: {}'.format(hours), 
        workyears_text='TotalWorkingYears: {}'.format(workyears),
        balance_text='WorkLifeBalance: {}'.format(balance),
        role_text = 'YearsInCurrentRole: {}'.format(role),
        promotion_text = 'YearsSinceLastPromotion: {}'.format(promotion),
        mgr_text = 'YearsWithCurrManager: {}'.format(mgr)
        )
        
         # rendering the predicted result 
    # retrieving values from form
    # init_features = [float(x) for x in request.form.values()]
    # final_features = [np.array(init_features)]

    # prediction = model.predict(final_features) # making prediction

    # class_ = np.where(predictions == np.amax(predictions, axis=1))[1][0]

    # return render_template('machinelearning.html', prediction_text='Attrition Likelihood: {}'.format(prediction), pred=class_) # rendering the predicted result


@app.route('/past')
def past():
    print("/past")
    return render_template('past.html')


@app.route('/project')
def project():
    print("/project")
    return render_template('project.html')




if __name__ == "__main__":
    app.run(debug=True)
