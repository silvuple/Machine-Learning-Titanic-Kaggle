import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import RFECV


# Read titanic 'train' and 'test' DataFrames from pickle files.
train = pd.read_pickle('train.pkl')
test = pd.read_pickle('test.pkl')

# Create training/testing set.
all_features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 
                'Embarked', 'Last_name', 'Age_group', 
                'Relative_fare', 'Relative_age']
X = train.loc[:, all_features]
y = train['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y)
X_predict = test.loc[:, all_features]

# Select/eliminate feature with recursive feature elimination (RFECV)
logreg = LogisticRegression()
rfecv = RFECV(estimator=logreg, cv=5)

# Get cross_validation mean accuracy scores.
mean_cv_score = cross_val_score(rfecv, X, y, cv=5).mean()
print("cross_val accuracy score is", mean_cv_score)

# Fit the model.
rfecv.fit(X, y)

print("Optimal number of features: {}".format(rfecv.n_features_))
features_ranking = pd.DataFrame({'Feature':X.columns, 'Ranking':rfecv.ranking_})
print("Selected Features with Ranking = 1:\n", features_ranking)
print("the mean score is:", rfecv.grid_scores_.mean())

# Plot number of features VS. cross-validation scores.
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
test.loc[:, ['PassengerId','Survived']].to_csv('submission_lgrg.csv', index=False)
