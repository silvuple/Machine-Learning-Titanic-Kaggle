import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, GridSearchCV


# Read titanic 'train' and 'test' DataFrames from pickle files.
train = pd.read_pickle('train.pkl')
test = pd.read_pickle('test.pkl')

# Create training/testing set.
all_features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 
                'Fare', 'Embarked', 'Last_name', 'Age_group', 
                'Relative_fare', 'Relative_age']
select_features = ['Pclass', 'Sex', 'Age']
X = train.loc[:, select_features]
y = train['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y)
X_predict = test.loc[:, select_features]


# Run classification model - Support Vector Classification.
# Create classifier instance and fit the model on training data.
svc = SVC()
svc.fit(X_train, y_train)

# Get mean accuracy and cross_validation mean accuracy scores.
accuracy_score = svc.score(X_test, y_test)
mean_cv_score = cross_val_score(svc, X, y, cv=5).mean()
print("train_test_split svc score is", accuracy_score)
print("cross_val accuracy svc score is", mean_cv_score)

# Search for best C and gamma parameters for scv.
C_options = [2**x for x in range(-2, 15)]
gamma_options = [2**x for x in range(-5, 2)]
param_grid = dict(C=C_options, gamma=gamma_options)
grid = GridSearchCV(svc, param_grid, cv=5)
grid.fit(X, y)
print("grid best score is", grid.best_score_)
print("grid best params are", grid.best_params_)

# Predict for 'test' data.
test['Survived'] = grid.predict(X_predict)

# Create csv submission file.
test.loc[:, ['PassengerId','Survived']].to_csv('submission_svc.csv', index=False)
