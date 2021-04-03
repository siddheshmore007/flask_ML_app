import numpy as np
import pickle
import pandas as pd




#Importing the dataset
iplds = pd.read_csv('ipl.csv')
columns_to_remove = ['mid', 'venue', 'batsman', 'bowler', 'striker', 'non-striker']
iplds.drop(labels = columns_to_remove, axis=1, inplace=True)


consistent_teams = ['Kolkata Knight Riders', 'Chennai Super Kings', 'Rajasthan Royals', 'Mumbai Indians', 'Kings XI Punjab',
                     'Royal Challengers Bangalore', 'Delhi Daredevils', 'Sunrisers Hyderabad']

iplds = iplds[(iplds['bat_team'].isin(consistent_teams)) & (iplds['bowl_team'].isin(consistent_teams))]

iplds = iplds[iplds['overs']>=5.0]

# Converting the column 'date' from string into datetime object
#from datetime import datetime
#iplds['date'] = iplds['date'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))


encoded_iplds = pd.get_dummies(data=iplds, columns=['bat_team', 'bowl_team'])

encoded_iplds = encoded_iplds[['date', 'bat_team_Chennai Super Kings', 'bat_team_Delhi Daredevils', 'bat_team_Kings XI Punjab',
              'bat_team_Kolkata Knight Riders', 'bat_team_Mumbai Indians', 'bat_team_Rajasthan Royals',
              'bat_team_Royal Challengers Bangalore', 'bat_team_Sunrisers Hyderabad',
              'bowl_team_Chennai Super Kings', 'bowl_team_Delhi Daredevils', 'bowl_team_Kings XI Punjab',
              'bowl_team_Kolkata Knight Riders', 'bowl_team_Mumbai Indians', 'bowl_team_Rajasthan Royals',
              'bowl_team_Royal Challengers Bangalore', 'bowl_team_Sunrisers Hyderabad',
              'overs', 'runs', 'wickets', 'runs_last_5', 'wickets_last_5', 'total']]

X = encoded_iplds.iloc[:, 1:-1].values
y = encoded_iplds.iloc[:, -1].values



from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#
# #
# #

# #
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)
regressor.fit(X_train, y_train)

filename = 'first-innings-score-rfr-model.pkl'
pickle.dump(regressor, open(filename, 'wb'))
