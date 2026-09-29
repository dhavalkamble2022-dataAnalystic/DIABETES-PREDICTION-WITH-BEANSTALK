# Diabetes Prediction

import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns


# Read the data file
data = pd.read_csv(
    r"C:\Users\HP\OneDrive\Documents\Desktop\Data Analystics\Logistic Implementation\DiabeticPrediction\Dataset\diabetes.csv"
)

data.head()

data.describe()

data.isnull().sum()

# There are a few misconceptions, like BMI cannot be zero,
# BP can't be zero, glucose and insulin can't be zero.

# Now replacing zero values with the mean of the column

data['BMI'] = data['BMI'].replace(0, data['BMI'].mean())

data['BloodPressure'] = data['BloodPressure'].replace(
    0, data['BloodPressure'].mean()
)

data['Glucose'] = data['Glucose'].replace(
    0, data['Glucose'].mean()
)

data['Insulin'] = data['Insulin'].replace(
    0, data['Insulin'].mean()
)

data['SkinThickness'] = data['SkinThickness'].replace(
    0, data['SkinThickness'].mean()
)


#Now we have deal with 0 values and data looks better , But there still are outliers present and some column

fig,ax=plt.subplots(figsize=(15,10))
sns.boxplot(data=data, width=0.5,ax=ax)
# plt.show()

data.head()

#segregate the dependdent and independent variable
X=data.drop(columns=['Outcome'])
y=data['Outcome']

#seperate dataset into train and test
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=0)
X_train.shape,X_test.shape

import pickle
#Standard scaling- Standardization
def scaler_standard(X_train,X_test):
  #Scaling the data
  scaler=StandardScaler()
  X_train_scaled=scaler.fit_transform(X_train)
  X_test_scaled=scaler.transform(X_test)

  #saving the model
  file=open('Model/standardScaler.pkl','wb')
  pickle.dump(scaler,file)
  file.close()

  return X_train_scaled,X_test_scaled

X_train_scaled,X_test_scaled=scaler_standard(X_train,X_test)
print(X_train_scaled)

log_reg=LogisticRegression()
log_reg.fit(X_train_scaled,y_train)

#Hyperparameter tunning
#GRIDSEARCH CV
from sklearn.model_selection import GridSearchCV
import numpy as np
import warnings
warnings.filterwarnings('ignore')
#parameter grid
parameter={ 'penalty':['l1','l2'], 'C':np.logspace(-3,3,7),'solver':['newton-cg','lbfgs','liblinear']}

logreg = LogisticRegression()

clf = GridSearchCV(
    logreg,
    param_grid=parameter,
    scoring='accuracy',
    cv=10
)

clf.fit(X_train_scaled, y_train)

clf.best_params_
clf.best_score_

y_pred=clf.predict(X_test_scaled)

conf_mat=confusion_matrix(y_test,y_pred)
conf_mat

true_positive = conf_mat[0][0]
false_positive = conf_mat[0][1]
false_negative = conf_mat[1][0]
true_negative = conf_mat[1][1]

Accuracy=(true_positive+true_negative)/(true_positive+true_negative+false_positive+false_negative)
print(Accuracy)

Precision=(true_positive)/(true_positive+false_positive)
print(Precision)

Recall=(true_positive)/(true_positive+false_negative)
print(Recall)

F1_Score=2*(Recall*Precision)/(Recall+Precision)
print(F1_Score)


import pickle
file=open('Model/modelForPrediction.pkl','wb')
pickle.dump(log_reg,file)
file.close()

