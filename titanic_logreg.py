import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
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
train.to_csv('check_train_after_process.csv')

# 2. Select features
# 2.2. Creating training/testing set.
all_features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Embarked', 'Last_name', 'Age_group', 'Relative_fare', 'Relative_age']
select_features = ['Sex', 'Relative_age', 'Relative_fare']
X = train[all_features]
y = train['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y)
X_predict = test[all_features]

# Create the RFE object and compute a cross-validated score.
logreg=LogisticRegression()
# The "accuracy" scoring is proportional to the number of correct
# classifications
rfecv = RFECV(estimator=logreg, step=1, cv=StratifiedKFold(2),
              scoring='accuracy')
rfecv.fit(X, y)
X_r = rfecv.transform(X)

print(X_r[:5])
print("Optimal number of features : %d" % rfecv.n_features_)

# Plot number of features VS. cross-validation scores
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (nb of correct classifications)")
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
plt.show()



#  3. Run classification model - Logistic Regression.

# fit the model
logreg=LogisticRegression()
logreg.fit(X_train, y_train)

# get mean accuracy and cross_validation mean accuracy scores
logreg_score = logreg.score(X_test, y_test)
print("mean accuracy score is ", logreg_score, '\n')

logreg_cross_val_score = cross_val_score(logreg, X, y, cv=10, scoring='accuracy').mean()
print("cross_validation mean accuracy score is ", logreg_cross_val_score, '\n')

# train on all train data
logreg.fit(X, y)

# predict for test data
test['Survived'] = logreg.predict(X_predict)

# create csv submission file
test[['PassengerId','Survived']].to_csv('submission_lgrg.csv', index=False)
