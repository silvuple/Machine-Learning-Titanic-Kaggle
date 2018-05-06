import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import RFE
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold

# Read titanic train and test csv files into pandas DataFrame.
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

print(train.head())

# 1. Process the data
# 1.1. Create new column 'Last_name' based on 'Name' column.
train['Last_name'] = train['Name'].str.partition(pat=',').iloc[:, 0]
test['Last_name'] = test['Name'].str.partition(pat=',').iloc[:, 0]


# 1.2. Remove 'PassengerId', 'Name' columns as insignificant and 'Cabin' column as insufficient data. 
# 'PassengerId' column in test set is dropped later for prediction only as it is needed for submission file
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
        if age>=value[0] and age<=value[1]:
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

train.Ticket=le.transform(Ticket_values_from_train)
test.Ticket=le.transform(Ticket_values_from_test)


# 1.8. Encode non-numeric 'Last_name' column  to numeric values.
Last_name_values_from_train = train.Last_name.values.tolist()
Last_name_values_from_test = test.Last_name.values.tolist()
Last_name_values = Last_name_values_from_train + Last_name_values_from_test
Last_name_unique_values = list(set(Last_name_values))

le2 = LabelEncoder()
le2.fit(Last_name_unique_values)

train.Last_name=le2.transform(Last_name_values_from_train)
test.Last_name=le2.transform(Last_name_values_from_test)


# 1.9. Create new column 'Relative_fare' from 'Fare' column.
avg_fare = train.Fare.mean()
train['Relative_fare'] = train.Fare.apply(lambda x: x/avg_fare)
test['Relative_fare'] = test.Fare.apply(lambda x: x/avg_fare)


# 1.10. Create new column 'Relative_age' from 'Age' column.
avg_age = train.Age.mean()
train['Relative_age'] = train.Age.apply(lambda x: x/avg_fare)
test['Relative_age'] = test.Age.apply(lambda x: x/avg_fare)


print("TRAIN AFTER NON_NUMERICAL HANDLING")
print(train.head())
print(train.dtypes)

# 1.11. Convert 'Age' and 'Fare' columns to int types.
train['Age'] = train.Age.astype('int64')
test['Age'] = test.Age.astype('int64')
train['Fare'] = train.Fare.astype('int64')
test['Fare'] = test.Fare.astype('int64')

# 1.12. Round 'Relative_fare' and 'Relative_age' column values to 2 digits after decimal point.
train['Relative_fare'] = train.Relative_fare.apply(round, ndigits=2)
test['Relative_fare'] = test.Relative_fare.apply(round, ndigits=2)
train['Relative_age'] = train.Relative_age.apply(round, ndigits=2)
test['Relative_age'] = test.Relative_age.apply(round, ndigits=2)


print("TRAIN AFTER convert to INT64")
print(train.head())



# 2. Select features
# 2.2. Creating training/testing set.
all_features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Embarked', 'Last_name', 'Age_group', 'Relative_fare', 'Relative_age']
select_features = ['Pclass', 'Sex', 'SibSp', 'Parch', 'Fare', 'Age', 'Fare']
X = train[select_features]
y = train['Survived']
X_predict = test[select_features]


# 1.13. Create dummy variables for the feature columns using One-Hot Encoder
from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder(handle_unknown='ignore')

enc.fit(X)  
print(enc.n_values_)
print(enc.feature_indices_)
X = enc.transform(X).toarray()
X_predict = enc.transform(X_predict).toarray()


X_train, X_test, y_train, y_test = train_test_split(X, y)

# 4. RandomForestClassifier

# fit the model
rfc=RandomForestClassifier()
rfc.fit(X_train, y_train)

# get mean accuracy and cross_validation mean accuracy scores
accuracy_score = rfc.score(X_test, y_test)
print("mean accuracy score is ", accuracy_score, '\n')

rfc_cross_val_score = cross_val_score(rfc, X, y, cv=10, scoring='accuracy').mean()
print("mean cross_val accuracy rfc score is ", rfc_cross_val_score, '\n')

# Create the RFE object and compute a cross-validated score.
# The "accuracy" scoring is proportional to the number of correct classifications
rfecv = RFECV(estimator=rfc, step=1, cv=StratifiedKFold(2), scoring='accuracy')
rfecv.fit(X, y)
X = rfecv.transform(X)
X_predict = rfecv.transform(X_predict)
print("Optimal number of features : %d" % rfecv.n_features_)

### Plot number of features VS. cross-validation scores
##plt.figure()
##plt.xlabel("Number of features selected")
##plt.ylabel("Cross validation score (nb of correct classifications)")
##plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
##plt.show()

# check accuracy score after RFECV optimization
X_train, X_test, y_train, y_test = train_test_split(X, y)
rfc.fit(X_train, y_train)

# get mean accuracy and cross_validation mean accuracy scores
accuracy_score = rfc.score(X_test, y_test)
print("mean accuracy score is ", accuracy_score, '\n')

rfc_cross_val_score = cross_val_score(rfc, X, y, cv=10, scoring='accuracy').mean()
print("mean cross_val accuracy rfc score is ", rfc_cross_val_score, '\n')
# fit the rfc model on all the train data
rfc.fit(X, y)

# search for best  and best  with GridSearchCV:
n_estimators_range=list(range(3, 100))
##max_depth_range=list(range(1, 51, 10))
##max_features_range=list(range(1, 5))
param_grid = dict(n_estimators=n_estimators_range)
grid = GridSearchCV(rfc, param_grid, cv=5, scoring='accuracy')
grid.fit(X, y)
print("grid best score is :", grid.best_score_, '\n')
print("grid best params are :", grid.best_params_, '\n')

# predict for test data:
test['Survived'] = grid.predict(X_predict)
test[['PassengerId','Survived']].to_csv('submission_rfc.csv', index=False)
