#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


train=pd.read_csv('train_IQ.csv')
test=pd.read_csv('test_IQ.csv')


# In[3]:


print(test.shape)
print(train.shape)


# In[4]:


print(train.dtypes.value_counts())


# In[4]:


train


# In[5]:


print(train.info())


# In[6]:


train.loc[:,train.dtypes==np.object]


# In[6]:


train.drop(['Id','idhogar'],axis=1,inplace=True)


# In[7]:


train['dependency'].value_counts()


# In[7]:


def map(i):
    
    if i=='yes':
        return(float(1))
    elif i=='no':
        return(float(0))
    else:
        return(float(i))


# In[8]:


train['dependency']=train['dependency'].apply(map)


# In[9]:


train['dependency']


# In[10]:


for i in train.columns:
    a=train[i].dtype
    if a == 'object':
        print(i)


# In[11]:


train.info()


# In[12]:


train['edjefe']=train['edjefe'].apply(map)
train['edjefa']=train['edjefa'].apply(map)


# In[13]:


train.info()


# In[14]:


var_df=pd.DataFrame(np.var(train,0),columns=['variance'])
var_df.sort_values(by='variance').head(15)
print('Below are columns with variance 0.')
col=list((var_df[var_df['variance']==0]).index)
print(col)


# From above it is shown that all values of elimbasu5 is same so there is no variablity in dataset therefor we will drop this variable

# In[15]:


contingency_tab=pd.crosstab(train['r4t3'],train['hogar_total'])
Observed_Values=contingency_tab.values
import scipy.stats
b=scipy.stats.chi2_contingency(contingency_tab)
Expected_Values = b[3]
no_of_rows=len(contingency_tab.iloc[0:2,0])
no_of_columns=len(contingency_tab.iloc[0,0:2])
df=(no_of_rows-1)*(no_of_columns-1)
print("Degree of Freedom:-",df)
from scipy.stats import chi2
chi_square=sum([(o-e)**2./e for o,e in zip(Observed_Values,Expected_Values)])
chi_square_statistic=chi_square[0]+chi_square[1]
print("chi-square statistic:-",chi_square_statistic)
alpha=0.05
critical_value=chi2.ppf(q=1-alpha,df=df)
print('critical_value:',critical_value)
p_value=1-chi2.cdf(x=chi_square_statistic,df=df)
print('p-value:',p_value)
print('Significance level: ',alpha)
print('Degree of Freedom: ',df)
print('chi-square statistic:',chi_square_statistic)
print('critical_value:',critical_value)
print('p-value:',p_value)
if chi_square_statistic>=critical_value:
    print("Reject H0,There is a relationship between 2 categorical variables")
else:
    print("Retain H0,There is no relationship between 2 categorical variables")
    
if p_value<=alpha:
    print("Reject H0,There is a relationship between 2 categorical variables")
else:
    print("Retain H0,There is no relationship between 2 categorical variables")


# Therefore,variables ('r4t3','hogar_total') have relationship between them. For good result we can use any one of them.

# In[16]:


contingency_tab=pd.crosstab(train['tipovivi3'],train['v2a1'])
Observed_Values=contingency_tab.values
import scipy.stats
b=scipy.stats.chi2_contingency(contingency_tab)
Expected_Values = b[3]
no_of_rows=len(contingency_tab.iloc[0:2,0])
no_of_columns=len(contingency_tab.iloc[0,0:2])
df=(no_of_rows-1)*(no_of_columns-1)
print("Degree of Freedom:-",df)
from scipy.stats import chi2
chi_square=sum([(o-e)**2./e for o,e in zip(Observed_Values,Expected_Values)])
chi_square_statistic=chi_square[0]+chi_square[1]
print("chi-square statistic:-",chi_square_statistic)
alpha=0.05
critical_value=chi2.ppf(q=1-alpha,df=df)
print('critical_value:',critical_value)
p_value=1-chi2.cdf(x=chi_square_statistic,df=df)
print('p-value:',p_value)
print('Significance level: ',alpha)
print('Degree of Freedom: ',df)
print('chi-square statistic:',chi_square_statistic)
print('critical_value:',critical_value)
print('p-value:',p_value)
if chi_square_statistic>=critical_value:
    print("Reject H0,There is a relationship between 2 categorical variables")
else:
    print("Retain H0,There is no relationship between 2 categorical variables")
    
if p_value<=alpha:
    print("Reject H0,There is a relationship between 2 categorical variables")
else:
    print("Retain H0,There is no relationship between 2 categorical variables")


# Therefore,variables ('tipovivi3','v2a1') have relationship between them. For good result we can use any one of them

# In[17]:


contingency_tab=pd.crosstab(train['v18q'],train['v18q1'])
Observed_Values=contingency_tab.values
import scipy.stats
b=scipy.stats.chi2_contingency(contingency_tab)
Expected_Values = b[3]
no_of_rows=len(contingency_tab.iloc[0:2,0])
no_of_columns=len(contingency_tab.iloc[0,0:2])
df=(no_of_rows-1)*(no_of_columns-1)
print("Degree of Freedom:-",df)
from scipy.stats import chi2
chi_square=sum([(o-e)**2./e for o,e in zip(Observed_Values,Expected_Values)])
chi_square_statistic=chi_square[0]+chi_square[1]
print("chi-square statistic:-",chi_square_statistic)
alpha=0.05
critical_value=chi2.ppf(q=1-alpha,df=df)
print('critical_value:',critical_value)
p_value=1-chi2.cdf(x=chi_square_statistic,df=df)
print('p-value:',p_value)
print('Significance level: ',alpha)
print('Degree of Freedom: ',df)
print('chi-square statistic:',chi_square_statistic)
print('critical_value:',critical_value)
print('p-value:',p_value)
if chi_square_statistic>=critical_value:
    print("Reject H0,There is a relationship between 2 categorical variables")
else:
    print("Retain H0,There is no relationship between 2 categorical variables")
    
if p_value<=alpha:
    print("Reject H0,There is a relationship between 2 categorical variables")
else:
    print("Retain H0,There is no relationship between 2 categorical variables")


# Therefore,variables ('v18q','v18q1') have relationship between them. For good result we can use any one of them.

# *Therefore, there is bias in our dataset.*

# In[18]:


train.drop('r4t3',axis=1,inplace=True)


# In[19]:


train.parentesco1.value_counts()


# In[20]:


pd.crosstab(train['edjefa'],train['edjefe'])


# *Above cross tab shows 0 male head and 0 female head which implies that there are 435 families with no family head.*

# In[21]:


train.isna().sum().value_counts()


# In[22]:


train['Target'].isna().sum()


# There are no null values in Target variable. Now lets proceed further and identify and fillna of other variable.

# In[23]:


float_col=[]
for i in train.columns:
    a=train[i].dtype
    if a == 'float64':
        float_col.append(i)
print(float_col)


# In[24]:


train[float_col].isna().sum()


# In[25]:


train['v18q1'].value_counts()


# In[26]:


pd.crosstab(train['tipovivi1'],train['v2a1'])


# In[27]:


pd.crosstab(train['v18q1'],train['v18q'])


#  we can drop a column tipovivi3,v18q

# In[28]:


train['v2a1'].fillna(0,inplace=True)
train['v18q1'].fillna(0,inplace=True)


# In[29]:


train.drop(['tipovivi3', 'v18q','rez_esc','elimbasu5'],axis=1,inplace=True)


# In[30]:


train['meaneduc'].fillna(np.mean(train['meaneduc']),inplace=True)
train['SQBmeaned'].fillna(np.mean(train['SQBmeaned']),inplace=True)
print(train.isna().sum().value_counts())


# In[31]:


int_col=[]
for i in train.columns:
    a=train[i].dtype
    if a == 'int64':
        int_col.append(i)
print(int_col)


# In[32]:


train[int_col].isna().sum().value_counts()


# Now there is no null value in our datset.

# In[33]:


train.Target.value_counts()


# In[34]:


Poverty_level=train[train['v2a1'] !=0]


# In[35]:


Poverty_level.shape


# In[36]:


poverty_level=Poverty_level.groupby('area1')['v2a1'].apply(np.median)


# In[37]:


poverty_level


# For rural area level if people paying rent less than 8000 is under poverty level.
# For Urban area level if people paying rent less than 140000 is under poverty level.

# In[38]:


def povert(x):
    if x<8000:
        return('Below poverty level')
    
    elif x>140000:
        return('Above poverty level')
    elif x<140000:
        return('Below poverty level: Ur-ban ; Above poverty level : Rural ')


# In[39]:


c=Poverty_level['v2a1'].apply(povert)


# In[40]:


c.shape


# In[41]:


pd.crosstab(c,Poverty_level['area1'])


# Rural :
# 
# Above poverty level= 445
# 
# Urban :
# 
# Above poverty level =1103
# 
# Below poverty level=1081

# In[42]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


# In[43]:


X_data=train.drop('Target',axis=1)
Y_data=train.Target


# In[45]:


X_data_col=X_data.columns


# In[46]:


from sklearn.preprocessing import StandardScaler
SS=StandardScaler()
X_data_1=SS.fit_transform(X_data)
X_data_1=pd.DataFrame(X_data_1,columns=X_data_col)


# In[48]:


X_train,X_test,Y_train,Y_test=train_test_split(X_data_1,Y_data,test_size=0.25,stratify=Y_data,random_state=0)


# In[50]:


from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

rfc=RandomForestClassifier(random_state=0)
parameters={'n_estimators':[10,50,100,300],'max_depth':[3,5,10,15]}
grid=zip([rfc],[parameters])

best_=None

for i, j in grid:
    a=GridSearchCV(i,param_grid=j,cv=3,n_jobs=1)
    a.fit(X_train,Y_train)
    if best_ is None:
        best_=a
    elif a.best_score_>best_.best_score_:
        best_=a
        
        
print ("Best CV Score",best_.best_score_)
print ("Model Parameters",best_.best_params_)
print("Best Estimator",best_.best_estimator_)


# In[51]:


RFC=best_.best_estimator_
Model=RFC.fit(X_train,Y_train)
pred=Model.predict(X_test)


# In[52]:


print('Model Score of train data : {}'.format(Model.score(X_train,Y_train)))
print('Model Score of test data : {}'.format(Model.score(X_test,Y_test)))


# In[53]:


Important_features=pd.DataFrame(Model.feature_importances_,X_data_col,columns=['feature_importance'])


# In[54]:


Top50Features=Important_features.sort_values(by='feature_importance',ascending=False).head(50).index


# In[55]:


Top50Features


# In[56]:


for i in Top50Features:
    if i not in X_data_col:
        print(i)


# In[57]:


X_data_Top50=X_data[Top50Features]


# In[58]:


X_train,X_test,Y_train,Y_test=train_test_split(X_data_Top50,Y_data,test_size=0.25,stratify=Y_data,random_state=0)


# In[59]:


Model_1=RFC.fit(X_train,Y_train)
pred=Model_1.predict(X_test)


# In[60]:


from sklearn.metrics import confusion_matrix,f1_score,accuracy_score


# In[61]:


confusion_matrix(Y_test,pred)


# In[62]:


f1_score(Y_test,pred,average='weighted')


# In[63]:


accuracy_score(Y_test,pred)


# In[64]:


test.drop('r4t3',axis=1,inplace=True)
test.drop(['Id','idhogar'],axis=1,inplace=True)
test['dependency']=test['dependency'].apply(map)
test['edjefe']=test['edjefe'].apply(map)
test['edjefa']=test['edjefa'].apply(map)


# In[65]:


test['v2a1'].fillna(0,inplace=True)
test['v18q1'].fillna(0,inplace=True)


# In[66]:


test.drop(['tipovivi3', 'v18q','rez_esc','elimbasu5'],axis=1,inplace=True)


# In[67]:


train['meaneduc'].fillna(np.mean(train['meaneduc']),inplace=True)
train['SQBmeaned'].fillna(np.mean(train['SQBmeaned']),inplace=True)


# In[68]:


test_data=test[Top50Features]


# In[69]:


test_data.isna().sum().value_counts()


# In[70]:


test_data.SQBmeaned.fillna(np.mean(test_data['SQBmeaned']),inplace=True)


# In[71]:


test_data.meaneduc.fillna(np.mean(test_data['meaneduc']),inplace=True)


# In[72]:


Test_data_1=SS.fit_transform(test_data)
X_data_1=pd.DataFrame(Test_data_1)


# In[73]:


test_prediction=Model_1.predict(test_data)


# In[74]:


test_prediction


# Above is our prediction for test data.*

# Using RandomForest Classifier we can predict test_data with accuracy of 90%.

# In[ ]:




