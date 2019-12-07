# --------------
# import the libraries
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Code starts here
df = pd.read_csv(path)
print(df.head())

df_copy = df.copy()
df_copy.drop('insuranceclaim',axis=1,inplace=True)

X = df_copy
print(X)
y = df['insuranceclaim']
print(y)

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state = 6)

# Code ends here


# --------------
import matplotlib.pyplot as plt


# Code starts here
plt.boxplot(X_train['bmi'])
q_value = X_train['bmi'].quantile(0.95)
plt.show()

print(y_train.value_counts())
# Code ends here


# --------------
# Code starts here
import seaborn as sns

relation = X_train.corr()
print(relation)

sns.pairplot(X_train)

# Code ends here


# --------------
import seaborn as sns
import matplotlib.pyplot as plt

# code starts here

# store categorical variable
cols = ['children','sex','region','smoker']
        


# create subplot 

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(20,20))

# create loop for plotting countplot

for i in range(0,2):
    for j in range(0,2):
             col=cols[i*2 + j]
             sns.countplot(x=X_train[col], hue=y_train, ax=axes[i,j])

# Code ends here




# --------------
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# parameters for grid search
parameters = {'C':[0.1,0.5,1,5]}

# Code starts here
lr = LogisticRegression(random_state=9)

grid = GridSearchCV(lr,parameters)

grid.fit(X_train,y_train)
y_pred = grid.predict(X_test)
print(y_pred)

accuracy = accuracy_score(y_test,y_pred)
print(accuracy)
# Code ends here


# --------------
from sklearn.metrics import roc_auc_score
from sklearn import metrics

# Code starts here
score = roc_auc_score(y_test, y_pred)

y_pred_proba = grid.predict_proba(X_test)[:,1]

fpr, tpr, thresholds = metrics.roc_curve(y_test,y_pred)

roc_auc =  roc_auc_score(y_test, y_pred_proba)

plt.plot(fpr,tpr,label="Logistic model, auc="+str(roc_auc))
# Code ends here


