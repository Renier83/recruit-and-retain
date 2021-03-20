import json
import pandas as pd
import pymysql
from sqlalchemy import create_engine
import sqlalchemy
from flask import Flask, request, render_template, jsonify
import os
import pickle
import numpy as np
import tensorflow as tf
import keras
from keras.models import load_model
from keras.models import Sequential


# Initialize Flask application
app = Flask(__name__)

# Use pickle to load in the pre-trained Logistic Regression model.
model = pickle.load(open('model.pkl', 'rb')) # loading the trained model
# X_test = pd.read_csv('model_X_test.csv')
# y_test = pd.read_csv('model_y_test.csv')
# y_test = y_test['0']

# model_score = model.score(X_test, y_test)
# print(model_score)

#tensorflow
predmodel = load_model('neural_model.h5')

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
        otimeno = request.form['OverTime_No']
        #otimeyes = request.form['OverTime_Yes']
        numco = request.form['NumCompaniesWorked']
        yearsco = request.form['YearsAtCompany']

        #set Overtime_Yes
        if otimeno == '1':
            otimeyes = '0'
        else: otimeyes = '1'

        #input variables for Logistic Regression
        input_variables = pd.DataFrame([[age, distance, level, satisfaction,
                    income, hours, workyears,balance,role,promotion,mgr]],
                    columns=['Age', 'DistanceFrormHome', 'JobLevel', 'JobSatisfaction',
                    'MonthlyIncome', 'StandadHours', 'TotalWorkingYears','WorkLifeBalance',
                    'YearsInCurrentRole','YearsSinceLastPromotion','YearsWithCurrManager'],
                    dtype=float)
            
        #LogisticRegressionPrediction
        prediction = model.predict(input_variables)[0]

                
        #input variables for Sequential
        input_variables2 = pd.DataFrame([[income, workyears, age, numco,
                distance, yearsco, otimeno, otimeyes]],
                columns=['MonthlyIncome', 'TotalWorkingYears', 'Age', 'NumCompaniesWorked',
                'DistanceFrormHome', 'YearsAtCompany', 'OverTime_No', 'OverTime_Yes'],
                dtype=float)
                
              
        #SequentialRegressionPrediction
        if predmodel.predict(input_variables2)[0][1] >= .5:
            prediction2 = 'Yes'
        else: 
            prediction2 = 'No'

        
        #return Prediction
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
                'YearsWithCurrManager': mgr,
                'NumCompaniesWorked': numco,
                'YearsAtCompany': yearsco,
                'OverTime_No': otimeno,
                'OverTime_Yes': otimeyes
            }, result=prediction, result2=prediction2,

        prediction_text='Attrition Likelihood (Logistic Regression): {}'.format(prediction),  #returns the values entered
        #accuracy_text='test: {}'.format(accuracy),
        prediction2_text='Attrition Likelihood (Sequential Model): {}'.format(prediction2),
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
        mgr_text = 'YearsWithCurrManager: {}'.format(mgr),
        numco_text = 'NumCompaniesWorked: {}'.format(numco),
        yearsco_text = 'YearsAtCompany: {}'.format(yearsco),
        otimeno_text = 'OverTime_No: {}'.format(otimeno),
        otimeyes_text= 'OverTime_Yes: {}'.format(otimeyes)
        )
        
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
