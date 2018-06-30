# Machine-Learning-Titanic-Kaggle
### Predict survival on Titanic and get familiar with ML basics
[link to competition page](https://www.kaggle.com/c/titanic)

basic requirements:
*python 3
*pandas
*sklean

additional libraries/moduls may be required such as:
*matplotlib
*numpy

Each script file utilizes on of the SciKit Learn (sklearn) classiriers: 
* RandomForestClassifier
* LogisticRegression
* KNeighborsClassifier
* SVC (Support Vector Classification)

Different sklearn techniques used thoughout the scripts:
*RFE/RFECV (Feature ranking with recursive feature elimination and cross-validated selection of the best number of features)
*GridSearchCV (Exhaustive search over specified parameter values for an estimator)
*LabelEncoder (Encode labels with value, transform non-numerical labels to numerical labels)
*cross_val_score (Evaluate classifier score by cross-validation to eliminate overfitting)

Future tasks:
1.Try DecisionTreeClassifier
2.Work on Feature engineering
3.Try data normalization, try OneHotEncoder
4.Work on feature selection techniques

Current mean accuracy score ~0.77
