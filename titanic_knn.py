import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score


# Read titanic 'train' and 'test' DataFrames from pickle files.
train = pd.read_pickle('train.pkl')
test = pd.read_pickle('test.pkl')

# Create training/testing set.
all_features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 
                'Fare', 'Embarked', 'Last_name', 'Age_group', 
                'Relative_fare', 'Relative_age']
select_features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch']
X = train.loc[:, select_features]
y = train['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y)
X_predict = test.loc[:, select_features]

# Run classification model - K-Nearest Neighbors Classifier.
# Create classifier instance.
knn = KNeighborsClassifier()

# Fit the model on training data.
knn.fit(X_train, y_train)

# Get mean accuracy and cross_validation mean accuracy scores.
accuracy_score = knn.score(X_test, y_test)
mean_cv_score = cross_val_score(knn, X, y, cv=5).mean()
print("mean accuracy score is", accuracy_score)
print("cross_val accuracy score is", mean_cv_score)

# Get best number of neighbors (one with max accuracy score).
k_range = list(range(1, 31))
k_scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X, y, cv=5)
    k_scores.append(scores.mean())
best_k = k_scores.index(max(k_scores))+1
print("best number of neighbors is", best_k)

# Plot number of neighbors against the cross validation accuracy score.
plt.plot(k_range, k_scores)
plt.xlabel('value of k in knn')
plt.ylabel('cross_validated accuracy')
plt.show()

# Re-instantiate the classifier with the number of neighbors.
knn = KNeighborsClassifier(n_neighbors=best_k)
knn.fit(X_train, y_train)

# Get new mean accuracy and cross_validation mean accuracy scores.
print("new mean accuracy score is", knn.score(X_test, y_test))
print("new cross_validation score is", cross_val_score(knn, X, y, cv=5).mean())

# Train the model on all the train data.
knn.fit(X, y)

# Predict for 'test' data.
test['Survived'] = knn.predict(X_predict)

# Create csv submission file.
test.loc[:, ['PassengerId','Survived']].to_csv('submission_knn.csv', index=False)
