import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import GridSearchCV


# Read titanic 'train' and 'test' DataFrames from pickle files.
train = pd.read_pickle('train.pkl')
test = pd.read_pickle('test.pkl')

# Pre-select features.
all_features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 
                'Fare', 'Embarked', 'Last_name', 'Age_group', 
                'Relative_fare', 'Relative_age']
select_features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch']
X = train.loc[:, select_features]
y = train['Survived']
X_predict = test.loc[:, select_features]

### Create dummy variables for the feature columns using One-Hot Encoder.
##enc = OneHotEncoder(handle_unknown='ignore')
##enc.fit(X)  
##print(enc.n_values_)
##print(enc.feature_indices_)
##X = enc.transform(X).toarray()
##X_predict = enc.transform(X_predict).toarray()
### Code below needs to be adjusted if OneHotEncoding is used.

# Create train/test split.
X_train, X_test, y_train, y_test = train_test_split(X, y)

# Use RandomForestClassifier to build prediction model.
# Instantiate and fit the model.
rfc = RandomForestClassifier()
rfc.fit(X_train, y_train)

# Get mean accuracy and cross_validation mean accuracy scores.
accuracy_score = rfc.score(X_test, y_test)
cv_score = cross_val_score(rfc, X, y, cv=5).mean()
print("RFC mean accuracy score is", accuracy_score)
print("RFC mean cross_val accuracy score is", cv_score)

# Get feature importance (the higher, the more important).
features_importance = pd.DataFrame({'Feature':X.columns,
                                    'Importance':rfc.feature_importances_})
print("Feature importance:\n", features_importance)

# Search for best parameters with GridSearchCV.
param_grid = {'max_depth': [80, 110, None], 
              'max_features': [2, 'auto', None], 
              'min_samples_leaf': [1, 2, 4], 
              'min_samples_split': [2, 4], 
              'n_estimators': [100, 200, 1000]}
grid = GridSearchCV(rfc, param_grid, cv=5)
grid.fit(X, y)
print("grid best score is:", grid.best_score_)
print("grid best params are:", grid.best_params_)

# Predict for test data.
test['Survived'] = grid.predict(X_predict)

# Create csv submission file.
test.loc[:, ['PassengerId','Survived']].to_csv('submission_rfc.csv', index=False)
