from flask import Flask, jsonify, render_template, request
import pickle
import numpy as np
import sys
from MLR import X_train, X_test, y_train, y_test
from yellowbrick.regressor import PredictionError, ResidualsPlot


# print(X_train)
# print(y_train)
# print(X_test)
# print(y_test)
#filename = 'first-innings-score-mlrmodel.pkl'

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')


"""
def parameters():
    if request.method == 'POST':
        regression_model = request.form['regression-model']
        if regression_model == 'Multiple Linear Regression':
            filename = 'first-innings-score-mlr-model.pkl'

        elif regression_model == 'Random Forest Regression':
            filename = 'first-innings-score-rfr-model.pkl'

        return render_template('chooseParameters.html', regressor=pickle.load(open(filename, 'rb')))
"""


@app.route('/parameters', methods=['POST'])
def parameters():
    filename = 'first-innings-score-mlr-model.pkl'
    if request.method == 'POST':
        regression_model = request.form['regression-model']
        if regression_model == 'Multiple Linear Regression':
            filename = 'first-innings-score-mlr-model.pkl'

        elif regression_model == 'Random Forest Regression':
            filename = 'first-innings-score-rfr-model.pkl'

    return render_template('chooseParameters.html', file=filename)


# def currentRegressor(filename):
#     global new_regressor = pickle.load(open(filename, 'rb'))
#     return new_regressor


# regressor = currentRegressor.new

# # parameters()
# # regressor = parameters.regressor


@app.route('/predict', methods=["GET", "POST"])
def predict():
    filename = request.form['name']
    regressor = pickle.load(open(filename, 'rb'))

    temp_array = list()

    if request.method == 'POST':
        batting_team = request.form['batting-team']
        if batting_team == 'Chennai Super Kings':
            temp_array = temp_array + [1, 0, 0, 0, 0, 0, 0, 0]
        elif batting_team == 'Delhi Daredevils':
            temp_array = temp_array + [0, 1, 0, 0, 0, 0, 0, 0]
        elif batting_team == 'Kings XI Punjab':
            temp_array = temp_array + [0, 0, 1, 0, 0, 0, 0, 0]
        elif batting_team == 'Kolkata Knight Riders':
            temp_array = temp_array + [0, 0, 0, 1, 0, 0, 0, 0]
        elif batting_team == 'Mumbai Indians':
            temp_array = temp_array + [0, 0, 0, 0, 1, 0, 0, 0]
        elif batting_team == 'Rajasthan Royals':
            temp_array = temp_array + [0, 0, 0, 0, 0, 1, 0, 0]
        elif batting_team == 'Royal Challengers Bangalore':
            temp_array = temp_array + [0, 0, 0, 0, 0, 0, 1, 0]
        elif batting_team == 'Sunrisers Hyderabad':
            temp_array = temp_array + [0, 0, 0, 0, 0, 0, 0, 1]

        bowling_team = request.form['bowling-team']
        if bowling_team == 'Chennai Super Kings':
            temp_array = temp_array + [1, 0, 0, 0, 0, 0, 0, 0]
        elif bowling_team == 'Delhi Daredevils':
            temp_array = temp_array + [0, 1, 0, 0, 0, 0, 0, 0]
        elif bowling_team == 'Kings XI Punjab':
            temp_array = temp_array + [0, 0, 1, 0, 0, 0, 0, 0]
        elif bowling_team == 'Kolkata Knight Riders':
            temp_array = temp_array + [0, 0, 0, 1, 0, 0, 0, 0]
        elif bowling_team == 'Mumbai Indians':
            temp_array = temp_array + [0, 0, 0, 0, 1, 0, 0, 0]
        elif bowling_team == 'Rajasthan Royals':
            temp_array = temp_array + [0, 0, 0, 0, 0, 1, 0, 0]
        elif bowling_team == 'Royal Challengers Bangalore':
            temp_array = temp_array + [0, 0, 0, 0, 0, 0, 1, 0]
        elif bowling_team == 'Sunrisers Hyderabad':
            temp_array = temp_array + [0, 0, 0, 0, 0, 0, 0, 1]

        overs = float(request.form['overs'])
        runs = int(request.form['runs'])
        wickets = int(request.form['wickets'])
        runs_in_prev_5 = int(request.form['runs_in_prev_5'])
        wickets_in_prev_5 = int(request.form['wickets_in_prev_5'])

        temp_array = temp_array + [overs, runs,
                                   wickets, runs_in_prev_5, wickets_in_prev_5]

        data = np.array([temp_array])
        my_prediction = int(regressor.predict(data)[0])

        model = regressor
        visualizer_pe = PredictionError(model)
        visualizer_pe.fit(X_train, y_train)
        visualizer_pe.score(X_test, y_test)
        vpe = visualizer_pe.poof()

        return render_template('prediction.html', lower_limit=my_prediction-10, upper_limit=my_prediction+5, vpe=vpe)

    # return render_template('prediction.html', regressor=regressor)

    # if request.method == 'POST':

    #     batting_team = request.form['batting-team']
    #     if batting_team == 'Chennai Super Kings':
    #         temp_array = temp_array + [1, 0, 0, 0, 0, 0, 0, 0]
    #     elif batting_team == 'Delhi Daredevils':
    #         temp_array = temp_array + [0, 1, 0, 0, 0, 0, 0, 0]
    #     elif batting_team == 'Kings XI Punjab':
    #         temp_array = temp_array + [0, 0, 1, 0, 0, 0, 0, 0]
    #     elif batting_team == 'Kolkata Knight Riders':
    #         temp_array = temp_array + [0, 0, 0, 1, 0, 0, 0, 0]
    #     elif batting_team == 'Mumbai Indians':
    #         temp_array = temp_array + [0, 0, 0, 0, 1, 0, 0, 0]
    #     elif batting_team == 'Rajasthan Royals':
    #         temp_array = temp_array + [0, 0, 0, 0, 0, 1, 0, 0]
    #     elif batting_team == 'Royal Challengers Bangalore':
    #         temp_array = temp_array + [0, 0, 0, 0, 0, 0, 1, 0]
    #     elif batting_team == 'Sunrisers Hyderabad':
    #         temp_array = temp_array + [0, 0, 0, 0, 0, 0, 0, 1]

    #     bowling_team = request.form['bowling-team']
    #     if bowling_team == 'Chennai Super Kings':
    #         temp_array = temp_array + [1, 0, 0, 0, 0, 0, 0, 0]
    #     elif bowling_team == 'Delhi Daredevils':
    #         temp_array = temp_array + [0, 1, 0, 0, 0, 0, 0, 0]
    #     elif bowling_team == 'Kings XI Punjab':
    #         temp_array = temp_array + [0, 0, 1, 0, 0, 0, 0, 0]
    #     elif bowling_team == 'Kolkata Knight Riders':
    #         temp_array = temp_array + [0, 0, 0, 1, 0, 0, 0, 0]
    #     elif bowling_team == 'Mumbai Indians':
    #         temp_array = temp_array + [0, 0, 0, 0, 1, 0, 0, 0]
    #     elif bowling_team == 'Rajasthan Royals':
    #         temp_array = temp_array + [0, 0, 0, 0, 0, 1, 0, 0]
    #     elif bowling_team == 'Royal Challengers Bangalore':
    #         temp_array = temp_array + [0, 0, 0, 0, 0, 0, 1, 0]
    #     elif bowling_team == 'Sunrisers Hyderabad':
    #         temp_array = temp_array + [0, 0, 0, 0, 0, 0, 0, 1]

    #     overs = float(request.form['overs'])
    #     runs = int(request.form['runs'])
    #     wickets = int(request.form['wickets'])
    #     runs_in_prev_5 = int(request.form['runs_in_prev_5'])
    #     wickets_in_prev_5 = int(request.form['wickets_in_prev_5'])

    #     temp_array = temp_array + [overs, runs,
    #                                wickets, runs_in_prev_5, wickets_in_prev_5]

    #     data = np.array([temp_array])
    #     my_prediction = int(regressor.predict(data)[0])

    #     return render_template('prediction.html', lower_limit=my_prediction-10, upper_limit=my_prediction+5)


if __name__ == '__main__':
    app.run(debug=True)
