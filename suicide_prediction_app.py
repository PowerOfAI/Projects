# -*- coding: utf-8 -*-
"""
Created on Sun Oct 30 23:29:09 2022

@author: Pawan
"""

from flask import Flask, render_template, request
import pandas as pd

app = Flask(__name__)
@app.route('/', methods=['GET', 'POST'])
def home():
    return render_template("index.html")

@app.route('/prediction_on_observers_data', methods=['GET', 'POST'])
def prediction_on_observers_data():
    if request.method =='POST':
        file = request.files['file']
        data = pd.read_csv(file)
        df1 = pd.get_dummies(data.iloc[:,4:], drop_first=True)
        
        custom_rating=[]
        for i in df1.iloc[:,:14].mean(axis=1): #taking mean of critical Qs only
            if i > 0:
                custom_rating.append(.99) # just 1 yes in critical category will make with 100% probable
            else:
                custom_rating.append(i)
        
        df1['critical_score'] = custom_rating
        df1['sensitive_score'] = df1.mean(axis=1)
        df1['final_score'] = df1.iloc[:,[-1,-2]].max(axis=1)*100
        
        
        #df1["Suicidal_tendency_in _%"] = df1.mean(axis=1)*100
        new_data = data
        new_data["Suicidal_tendency_in _%"] = df1["final_score"]
        new_data = new_data.rename(columns={"Name of the person who is the subject of the survey":"Student Name"})
        new_data["Suicidal_tendency_in _%"]  = round(new_data["Suicidal_tendency_in _%"])                            
        return render_template('data.html', data = new_data.iloc[:,[2,-1]].to_html())

@app.route('/prediction_on_self_data', methods=['GET', 'POST'])
def prediction_on_self_data():
    if request.method =='POST':
        file2 = request.files['selffile']
        data2 = pd.read_csv(file2)
        df12 = pd.get_dummies(data2.iloc[:,2:], drop_first=True)                
        df12["Suicidal_tendency_in _%"] = df12.mean(axis=1)*100
        new_data2 = data2
        new_data2["Suicidal_tendency_in _%"] = df12["Suicidal_tendency_in _%"]
        new_data2 = new_data2.rename(columns={"Your Name":"Student Name"})
        new_data2["Suicidal_tendency_in _%"]  = round(new_data2["Suicidal_tendency_in _%"])                            
        return render_template('data2.html', data2 = new_data2.iloc[:,[1,-1]].to_html())


if __name__ == "__main__":
    app.run(debug=True)
        