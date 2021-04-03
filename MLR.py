import numpy as np
import pickle
import pandas as pd


#Importing the dataset
iplds = pd.read_csv('ipl.csv')
columns_to_remove = ['mid', 'venue', 'batsman', 'bowler', 'striker', 'non-striker']
iplds.drop(labels = columns_to_remove, axis=1, inplace=True)
iplds.head()

consistent_teams = ['Kolkata Knight Riders', 'Chennai Super Kings', 'Rajasthan Royals', 'Mumbai Indians', 'Kings XI Punjab',
                     'Royal Challengers Bangalore', 'Delhi Daredevils', 'Sunrisers Hyderabad']

iplds = iplds[(iplds['bat_team'].isin(consistent_teams)) & (iplds['bowl_team'].isin(consistent_teams))]

iplds = iplds[iplds['overs']>=5.0]

# Converting the column 'date' from string into datetime object
from datetime import datetime
iplds['date'] = iplds['date'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))


encoded_iplds = pd.get_dummies(data=iplds, columns=['bat_team', 'bowl_team'])

encoded_iplds = encoded_iplds[['date', 'bat_team_Chennai Super Kings', 'bat_team_Delhi Daredevils', 'bat_team_Kings XI Punjab',
              'bat_team_Kolkata Knight Riders', 'bat_team_Mumbai Indians', 'bat_team_Rajasthan Royals',
              'bat_team_Royal Challengers Bangalore', 'bat_team_Sunrisers Hyderabad',
              'bowl_team_Chennai Super Kings', 'bowl_team_Delhi Daredevils', 'bowl_team_Kings XI Punjab',
              'bowl_team_Kolkata Knight Riders', 'bowl_team_Mumbai Indians', 'bowl_team_Rajasthan Royals',
              'bowl_team_Royal Challengers Bangalore', 'bowl_team_Sunrisers Hyderabad',
              'overs', 'runs', 'wickets', 'runs_last_5', 'wickets_last_5', 'total']]




""" Splitting the datatset into Training set and Testing set """
X_train = encoded_iplds.drop(labels='total', axis=1)[encoded_iplds['date'].dt.year <= 2016]
X_test = encoded_iplds.drop(labels='total', axis=1)[encoded_iplds['date'].dt.year >= 2017]

y_train = encoded_iplds[encoded_iplds['date'].dt.year <= 2016]['total'].values
y_test = encoded_iplds[encoded_iplds['date'].dt.year >= 2017]['total'].values


X_train.drop(labels='date', axis=True, inplace=True)
X_test.drop(labels='date', axis=True, inplace=True)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)


# Creating a pickle file for the classifier
filename = 'first-innings-score-mlr-model.pkl'
pickle.dump(regressor, open(filename, 'wb'))

# y_pred = regressor.predict(X_test)
# np.set_printoptions(precision=2)
# print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))
