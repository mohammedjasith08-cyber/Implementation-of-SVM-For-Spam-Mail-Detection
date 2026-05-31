# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the packages.
2. Analyse the data.
3. Use modelselection and Countvectorizer to preditct the values.
4. Find the accuracy and display the result.
 

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: MOHAMMED JASITH J
RegisterNumber: 212225230180 
*/
import pandas as pd
data=pd.read_csv("spam.csv", encoding='Windows-1252')
data
data.shape
x=data['v2'].values
y=data['v1'].values
x.shape
y.shape
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2, random_state=0)
x_train
x_train.shape
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)
from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
acc=accuracy_score(y_test,y_pred)
acc
con=confusion_matrix(y_test,y_pred)
print(con)
cl=classification_report(y_test,y_pred)
print(cl)
```

## Output:
## DATA
<img width="1093" height="671" alt="image" src="https://github.com/user-attachments/assets/bea442cc-2f58-4c72-a4ff-8bc87f272d51" />
## CONFUSION MATRIX
<img width="182" height="91" alt="image" src="https://github.com/user-attachments/assets/e7d8c35b-912b-425b-88d2-035323b19b49" />
## ACCURACY
<img width="312" height="71" alt="image" src="https://github.com/user-attachments/assets/4c3dc195-6c8c-45ed-ba0b-1c45f23beafd" />
## CLASSIFICATION REPORT
<img width="832" height="322" alt="image" src="https://github.com/user-attachments/assets/216c2e4e-7d53-445c-ac95-a10e76116c84" />



## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
