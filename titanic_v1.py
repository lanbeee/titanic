
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
import sklearn
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


# In[6]:


df = pd.read_csv('train.csv')


# In[29]:


X_test = pd.read_csv('test.csv')
y_train = df['Survived'].copy()
X_train = df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Cabin', 'Embarked']].copy()
X_test = X_test[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Cabin', 'Embarked']]
X_train.Cabin.fillna('Z', inplace= True)
X_test.Cabin.fillna('Z', inplace= True)
X_train.Embarked.fillna('S', inplace = True)
X_train['Embarked'] = X_train['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)
X_test['Embarked'] = X_test['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)
X_train.Age.fillna(28.0, inplace= True)
X_test.Fare.fillna(X_test.Fare.median(), inplace= True)
X_test.Age.fillna(28.0, inplace= True)


# In[30]:


trcb = [i[0] for i in X_train['Cabin']]
tscb = [i[0] for i in X_test['Cabin']]
trcabdum = pd.get_dummies(trcb)[['A', 'B', 'C', 'D', 'E', 'F']]
tscabdum = pd.get_dummies(tscb)[['A', 'B', 'C', 'D', 'E', 'F']]
trcabdum.columns = ['C_' + i for i in trcabdum.columns ]
tscabdum.columns = ['C_' + i for i in tscabdum.columns ]
trse = pd.get_dummies(X_train.Sex)[['female']]
tsse = pd.get_dummies(X_test.Sex)[['female']]
X_train = pd.concat([X_train,trcabdum,trse], axis= 1)
X_test = pd.concat([X_test,tscabdum,tsse], axis= 1)
X_train.drop(['Sex','Cabin'], axis=1, inplace= True)
X_test.drop(['Sex','Cabin'] , axis=1, inplace= True)
X_train['FareBand'] = pd.qcut(X_train['Fare'], 4)
combine = [X_train, X_test]
for dataset in combine:
    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
    dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3
    dataset['Fare'] = dataset['Fare'].astype(int)
X_train.drop(['FareBand'], axis= 1, inplace= True)
for dataset in combine:    
    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[ dataset['Age'] > 64, 'Age'] = 4
    dataset['Age'] = dataset['Age'].astype(int)
# X_train.head()


# In[33]:


X_train['C_ABDF'] = X_train['C_A'] + X_train['C_B'] + X_train['C_D'] + X_train['C_F']
X_test['C_ABDF'] = X_test['C_A'] + X_test['C_B'] + X_test['C_D'] + X_test['C_F']
X_train['C_EC'] = X_train['C_E'] + X_train['C_C']
X_test['C_EC'] = X_test['C_E'] + X_test['C_C']
X_train.drop(['C_A','C_B','C_C', 'C_D','C_E','C_F'], axis=1, inplace= True)
X_test.drop(['C_A','C_B','C_C', 'C_D','C_E','C_F'], axis=1, inplace= True)
X_train['oreone'] = X_train['C_ABDF']*X_train['female']
X_test['oreone'] = X_test['C_ABDF']*X_test['female']
X_train['oretwo'] = X_train['C_EC']*X_train['female']
X_test['oretwo'] = X_test['C_EC']*X_test['female']


# In[36]:


clf2 = RandomForestClassifier()
param_grid2 = [{'n_estimators':[70,100,150], 'max_depth':[3,4,5, None], 'max_features':[10,9,11]}]
gs2 = GridSearchCV(clf2, param_grid2, cv = 4, scoring= 'roc_auc')


# In[38]:


pred2 = pd.concat([pd.Series(np.arange(892,1310)),pd.Series(gs2.best_estimator_.predict(X_test))], axis=1)
pred2.columns = ['PassengerId','Survived']
# pred2.to_csv('pred6.csv',index=False)

