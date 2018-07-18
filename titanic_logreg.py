import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import RFECV


# Read titanic train and test csv files into pandas DataFrame.
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# 1. Process the data
# 1.1. Create new column 'Last_name' based on 'Name' column.
train['Last_name'] = train['Name'].str.partition(pat=',').iloc[:, 0]
test['Last_name'] = test['Name'].str.partition(pat=',').iloc[:, 0]

# 1.2. Remove 'PassengerId', 'Name' columns as insignificant and 
# 'Cabin' column as it has insufficient data, 'PassengerId' is 
# irrelevant for prediction, but needed for submission file
train.drop(columns=['PassengerId', 'Name', 'Cabin'], inplace=True)
test.drop(columns=['Name', 'Cabin'], inplace=True)

# 1.3. Replace missing values with zero.
train.fillna(value=0, inplace=True)
test.fillna(value=0, inplace=True)

# 1.4. Add new column 'Age_group' based on 'Age' column.
def age_group_column_creator(age):
    age_groups = {0: (0, 0),
                  1: (0.01, 3.00),
                  2: (3.01, 16.00),
                  3: (16.01, 30.00),
                  4: (30.01, 50.00),
                  5: (50.01, 120.00)}
    for key, value in age_groups.items():
        if age >= value[0] and age <= value[1]:
            return key

train['Age_group'] = train.Age.apply(age_group_column_creator)
test['Age_group'] = test.Age.apply(age_group_column_creator)

# 1.5. Map non-numeric 'Sex' column to numeric values.
train['Sex'] = train.Sex.map({'male': 1, 'female': 0})
test['Sex'] = test.Sex.map({'male': 1, 'female': 0})

# 1.6. Map non-numeric 'Embarked' column to numeric values.
train['Embarked'] = train.Embarked.map({'Q': 1, 'S': 2, 'C': 3, 0: 0})
test['Embarked'] = test.Embarked.map({'Q': 1, 'S': 2, 'C': 3, 0: 0})

# 1.7. Encode non-numeric 'Ticket' column  to numeric values.
Ticket_values_from_train = train.Ticket.values.tolist()
Ticket_values_from_test = test.Ticket.values.tolist()
Ticket_values = Ticket_values_from_train + Ticket_values_from_test
Ticket_unique_values = list(set(Ticket_values))

le = LabelEncoder()
le.fit(Ticket_unique_values)
train.Ticket = le.transform(Ticket_values_from_train)
test.Ticket = le.transform(Ticket_values_from_test)

# 1.8. Encode non-numeric 'Last_name' column  to numeric values.
Last_name_values_from_train = train.Last_name.values.tolist()
Last_name_values_from_test = test.Last_name.values.tolist()
Last_name_values = Last_name_values_from_train + Last_name_values_from_test
Last_name_unique_values = list(set(Last_name_values))

le2 = LabelEncoder()
le2.fit(Last_name_unique_values)
train.Last_name = le2.transform(Last_name_values_from_train)
test.Last_name = le2.transform(Last_name_values_from_test)

# 1.9. Create new column 'Relative_fare' from 'Fare' column.
avg_fare = train.Fare.mean()
train['Relative_fare'] = train.Fare.apply(lambda x: round(x/avg_fare, 2))
test['Relative_fare'] = test.Fare.apply(lambda x: round(x/avg_fare, 2))

# 1.10. Create new column 'Relative_age' from 'Age' column.
avg_age = train.Age.mean()
train['Relative_age'] = train.Age.apply(lambda x: round(x/avg_age, 2))
test['Relative_age'] = test.Age.apply(lambda x: round(x/avg_age, 2))

# 1.11. Convert 'Age' and 'Fare' columns to int types.
train['Age'] = train.Age.astype('int64')
test['Age'] = test.Age.astype('int64')
train['Fare'] = train.Fare.astype('int64')
test['Fare'] = test.Fare.astype('int64')


# 2. Select features and train the model
# Creating training/testing set.
all_features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 
                'Embarked', 'Last_name', 'Age_group', 
                'Relative_fare', 'Relative_age']
X = train.loc[:, all_features]
y = train['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y)
X_predict = test.loc[:, all_features]

# Feature selection/elimination with RFECV
logreg = LogisticRegression()
rfecv = RFECV(estimator=logreg, cv=5)

# Get cross_validation mean accuracy scores
mean_cv_score = cross_val_score(rfecv, X, y, cv=5).mean()
print("cross_val accuracy score is", mean_cv_score)

# Fit the model
rfecv.fit(X, y)

print("Optimal number of features: {}".format(rfecv.n_features_))
features_ranking = pd.DataFrame({'Feature':X.columns, 'Ranking':rfecv.ranking_})
print("Selected Features with Ranking = 1:\n", features_ranking)
print("the mean score is:", rfecv.grid_scores_.mean())

# Plot number of features VS. cross-validation scores
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (nb of correct classifications)")
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
plt.show()

# Reduce X and X_predict sets to the selected features. 
X_new = rfecv.transform(X)
X_predict_new = rfecv.transform(X_predict)

# Search for best parameters with GridSearchCV.
param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
grid = GridSearchCV(logreg, param_grid, cv=5)
grid.fit(X_new, y)
print("grid best score is:", grid.best_score_)
print("grid best params are:", grid.best_params_)

# Predict for 'test' data
test['Survived'] = grid.predict(X_predict_new)

# Create csv submission file
test[['PassengerId','Survived']].to_csv('submission_lgrg.csv', index=False)
