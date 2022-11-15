# Implementation of Decision Tree Regressor Model for Predicting the Salary of the Employee

# AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
# EQUIPMENTS REQUIRED:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

# ALGORITHM:
1. Import the required packages.
2. Read the data set.
3. Apply label encoder to the non-numerical column inoreder to convert into numerical values.
4. Determine training and test data set.
5. Apply decision tree regression on to the dataframe and get the values of Mean square error, r2 and data prediction.

# PROGRAM:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: yashaswi mitta
RegisterNumber: 212221230062
*/
```

```
import pandas as pd
data=pd.read_csv("Salary.csv")
data.head()

data.info()

data.isnull().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()

data["Position"]=le.fit_transform(data["Position"])
data.head()

x=data[["Position","Level"]]
x.head()

y=data[["Salary"]]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.2,random_state=2)

from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)

from sklearn import metrics
mse=metrics.mean_squared_error(y_test, y_pred)
mse

r2=metrics.r2_score(y_test,y_pred)
r2

dt.predict([[5,6]])
```

# OUTPUT:
![image](https://user-images.githubusercontent.com/94619247/201926333-b3cbc987-43a2-48eb-ac0d-c36861e1fdd7.png)
![image](https://user-images.githubusercontent.com/94619247/201926377-efa3a037-a566-4857-8f4b-32dc9bce461e.png)
![image](https://user-images.githubusercontent.com/94619247/201926424-070a4974-64e7-4019-8ef6-e158eac5eee5.png)


# RESULT:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming. 
