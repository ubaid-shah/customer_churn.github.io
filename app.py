from flask import Flask, render_template, request
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)

model = pickle.load(open('churn.pkl', 'rb'))


@app.route('/')
def man():
    return render_template('home.html')


@app.route('/predict', methods=['POST', 'GET'])
def predict():
    CreditScore = request.form["CreditScore"]
    Age = request.form["Age"]
    Tenure = request.form["Tenure"]
    Balance = request.form["Balance"]
    NumOfProducts = request.form["NumOfProducts"]
    HasCrCard = request.form["HasCrCard"]
    IsActiveMember = request.form["IsActiveMember"]
    EstimatedSalary = request.form["EstimatedSalary"]
    Female = request.form["Female"]
    Male = request.form["Male"]
    arr = [[CreditScore, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary, Female, Male]]
    col = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary',
           'Female', 'Male']

    pred_df = pd.DataFrame(arr, columns=col)

    pred = model.predict(pred_df)

    # return render_template('result.html', data=pred[0])

    if pred[0] == 0:
        return render_template('result.html', data="The customer will not exit")
    else:
        return render_template('result.html', data="The Customer will exit")


if __name__ == "__main__":
    app.run(debug=True)
