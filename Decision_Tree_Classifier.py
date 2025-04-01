import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv('bank.csv',delimiter=';')
print(df.head())
print(df.isna().sum())
df.head()

print(df[df.duplicated()])

from sklearn.preprocessing import OrdinalEncoder
oe=OrdinalEncoder()
df['education']=oe.fit_transform(df[['education']])
print(df.head())

from sklearn.preprocessing import LabelEncoder
lbe=LabelEncoder()
df['job']=lbe.fit_transform(df['job'])
df['marital']=lbe.fit_transform(df['marital'])
df['default']=lbe.fit_transform(df['default'])
df['housing']=lbe.fit_transform(df['housing'])
df['loan']=lbe.fit_transform(df['loan'])
df['contact']=lbe.fit_transform(df['contact'])
df['month']=lbe.fit_transform(df['month'])
df['poutcome']=lbe.fit_transform(df['poutcome'])
print(df.head(10))

from sklearn.model_selection import train_test_split
y=df['y']
x=df.drop('y',axis=1)
print(x)

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=0)

from sklearn.tree import DecisionTreeClassifier
dc=DecisionTreeClassifier(criterion='gini')
dc.fit(x_train,y_train)

pred=dc.predict(x_test)

from sklearn.metrics import accuracy_score
print(accuracy_score(pred,y_test))

sample_input = {
    'age': 35,
    'job': 10,
    'marital': 2,
    'education': 2.0,
    'default': 0,
    'balance': 1200,
    'housing': 1,
    'loan': 1,
    'contact': 0,
    'day': 5,
    'month': 8,
    'duration': 300,
    'campaign': 1,
    'pdays': -1,
    'previous': 0,
    'poutcome': 0
}

sample_ip=pd.DataFrame([sample_input])
prediction=dc.predict(sample_ip)
print("Predicted outcome is is",prediction)
