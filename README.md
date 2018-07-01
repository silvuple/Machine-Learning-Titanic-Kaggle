# Machine-Learning-Titanic-Kaggle
### Predict survival on Titanic and get familiar with ML basics
[link to competition page](https://www.kaggle.com/c/titanic)

basic requirements:
- python 3
- pandas
- sklearn

additional libraries/moduls may be required such as:
* matplotlib
* numpy

Each script file utilizes on of the SciKit Learn (sklearn) classiriers: 
1. RandomForestClassifier
2. LogisticRegression
3. KNeighborsClassifier
4. SVC (Support Vector Classification)
5. DecisionTreeClassifier

Different sklearn techniques used thoughout the scripts:
* RFE/RFECV (Feature ranking with recursive feature elimination and cross-validated selection of the best number of features)
* GridSearchCV (Exhaustive search over specified parameter values for an estimator)
* LabelEncoder (Encode labels with value, transform non-numerical labels to numerical labels)
* cross_val_score (Evaluate classifier score by cross-validation to eliminate overfitting)

Future tasks:
* Work on Feature engineering
* Try data normalization, try OneHotEncoder
* Work on feature selection techniques

Current mean accuracy score ~0.77
