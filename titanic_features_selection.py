import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Read titanic train and test csv files into pandas DataFrame.
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# Create new column 'Last_name' based on 'Name' column.
train['Last_name'] = train['Name'].str.partition(pat=',').iloc[:, 0]
test['Last_name'] = test['Name'].str.partition(pat=',').iloc[:, 0]

# Remove 'PassengerId', 'Name' columns as insignificant. 
# Remove 'Cabin' column as it has insufficient data.
# Leave 'PassengerId' in 'test' dataFrame because it is needed for
# submission file
train.drop(columns=['PassengerId', 'Name', 'Cabin'], inplace=True)
test.drop(columns=['Name', 'Cabin'], inplace=True)

# Replace missing values with zero.
train.fillna(value=0, inplace=True)
test.fillna(value=0, inplace=True)

# Create new column 'Age_group' based on 'Age' column.
def age_group(age):
    age_groups = {0: (0, 0),
                  1: (0.01, 3.00),
                  2: (3.01, 16.00),
                  3: (16.01, 30.00),
                  4: (30.01, 50.00),
                  5: (50.01, 120.00)}
    for key, value in age_groups.items():
        if age >= value[0] and age <= value[1]:
            return key
train['Age_group'] = train.Age.apply(age_group)
test['Age_group'] = test.Age.apply(age_group)

# Map non-numeric 'Sex' column to numeric values.
train['Sex'] = train.Sex.map({'male': 1, 'female': 0})
test['Sex'] = test.Sex.map({'male': 1, 'female': 0})

# Map non-numeric 'Embarked' column to numeric values.
train['Embarked'] = train.Embarked.map({'Q': 1, 'S': 2, 'C': 3, 0: 0})
test['Embarked'] = test.Embarked.map({'Q': 1, 'S': 2, 'C': 3, 0: 0})

# Encode non-numeric 'Ticket' column to numeric values.
ticket_values_train = train.Ticket.values.tolist()
ticket_values_test = test.Ticket.values.tolist()
ticket_values_all = ticket_values_train + ticket_values_test
ticket_values_unique = list(set(ticket_values_all))

le = LabelEncoder()
le.fit(ticket_values_unique)
train.Ticket = le.transform(ticket_values_train)
test.Ticket = le.transform(ticket_values_test)

# Encode non-numeric 'Last_name' column to numeric values.
last_name_values_train = train.Last_name.values.tolist()
last_name_values_test = test.Last_name.values.tolist()
last_name_values_all = last_name_values_train + last_name_values_test
last_name_values_unique = list(set(last_name_values_all))

le2 = LabelEncoder()
le2.fit(last_name_values_unique)
train.Last_name = le2.transform(last_name_values_train)
test.Last_name = le2.transform(last_name_values_test)

# Create new column 'Relative_fare' from 'Fare' column.
avg_fare = train.Fare.mean()
train['Relative_fare'] = train.Fare.apply(lambda x: round(x/avg_fare, 2))
test['Relative_fare'] = test.Fare.apply(lambda x: round(x/avg_fare, 2))

# Create new column 'Relative_age' from 'Age' column.
avg_age = train.Age.mean()
train['Relative_age'] = train.Age.apply(lambda x: round(x/avg_age, 2))
test['Relative_age'] = test.Age.apply(lambda x: round(x/avg_age, 2))

# Convert 'Age' and 'Fare' columns to int types.
train['Age'] = train.Age.astype('int64')
test['Age'] = test.Age.astype('int64')
train['Fare'] = train.Fare.astype('int64')
test['Fare'] = test.Fare.astype('int64')

# Pickle both DataFrames.
train.to_pickle('train.pkl')
test.to_pickle('test.pkl')
