import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import preprocessing

#Insert your csv file here
data = pd.read_csv('.csv')

# If feature columns contains strings; change them into binary:
# If no strings are included, remove this line
data = pd.get_dummies(data, columns=['string_columns_here'])

#Put your feature columns here
X = data['feature_columns'].values

#Put your target variable here
y = data['target_column'].values

#Split into training and testing set, change test_size and random_state to your own wishing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.0, random_state=0 )

decision_tree = DecisionTreeClassifier()
decision_tree = decision_tree.fit(X_train,y_train)

#Prediction
y_pred = decision_tree.predict(X_test)

#Get model accuracy:
print("Model Accuracy: ", accuracy_score(y_test, y_pred))