# --------------
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Code starts here
df = pd.read_csv(path)

print(df.head())

print(df.info())

df['INCOME'] = df.INCOME.str.replace('$','')
df['INCOME'] = df.INCOME.str.replace(',','')
df.sort_values('INCOME')

df['HOME_VAL'] = df.HOME_VAL.str.replace('$','')
df['HOME_VAL'] = df.HOME_VAL.str.replace(',','')
df.sort_values('HOME_VAL')

df['BLUEBOOK'] = df.BLUEBOOK.str.replace('$','')
df['BLUEBOOK'] = df.BLUEBOOK.str.replace(',','')
df.sort_values('BLUEBOOK')

df['OLDCLAIM'] = df.OLDCLAIM.str.replace('$','')
df['OLDCLAIM'] = df.OLDCLAIM.str.replace(',','')
df.sort_values('OLDCLAIM')

df['CLM_AMT'] = df.CLM_AMT.str.replace('$','')
df['CLM_AMT'] = df.CLM_AMT.str.replace(',','')
df.sort_values('CLM_AMT')

df_copy = df.copy()
df_copy.drop('CLAIM_FLAG',1,inplace=True)

X = df_copy
y = df['CLAIM_FLAG']

count = y.value_counts()

X_train,X_test,y_train,y_test = train_test_split(X, y, test_size = 0.3, random_state = 6)
# Code ends here


# --------------
# Code starts here
convert_dict = {'INCOME': float, 'HOME_VAL': float, 'BLUEBOOK': float, 
'OLDCLAIM': float, 'CLM_AMT': float}

X_train = X_train.astype(convert_dict)
X_test = X_test.astype(convert_dict)

print('X_train', X_train.isnull().sum())
print('X_test', X_test.isnull().sum())

# Code ends here


# --------------
# Code starts here
X_train.dropna(0, subset=['YOJ', 'OCCUPATION'], inplace=True)
X_test.dropna(0, subset=['YOJ', 'OCCUPATION'], inplace=True)


y_train=y_train[X_train.index]
y_test=y_test[X_test.index]


cols = ['AGE','CAR_AGE','INCOME','HOME_VAL']

[X_train[col].fillna(X_train[col].mean(), inplace=True) for col in cols]
[X_test[col].fillna(X_test[col].mean(), inplace=True) for col in cols]

# Code ends here


# --------------
from sklearn.preprocessing import LabelEncoder
columns = ["PARENT1","MSTATUS","GENDER","EDUCATION","OCCUPATION","CAR_USE","CAR_TYPE","RED_CAR","REVOKED"]

# Code starts here
for col in columns :
    le = LabelEncoder()
    X_train[col] = le.fit_transform(X_train[col].astype(str))
    X_test[col]  = le.transform(X_test[col].astype(str))

# Code ends here



# --------------
from sklearn.metrics import precision_score 
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression



# code starts here 
model = LogisticRegression(random_state = 6)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
score = accuracy_score(y_test,y_pred)

print('Score',score)

# Code ends here


# --------------
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

# code starts here

smote = SMOTE(random_state=9)
X_train, y_train  = smote.fit_sample(X_train, y_train)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Code ends here


# --------------
# Code Starts here
model = LogisticRegression(random_state = 6)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
score = accuracy_score(y_test,y_pred)

print('Score',score)

# Code ends here


