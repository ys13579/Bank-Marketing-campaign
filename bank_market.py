import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


df=pd.read_csv("C:\\Users\\asus\\Downloads\\bank-additional\\bank_additional_full.csv")
df.head()

df.describe()

####Age######
sns.countplot('age', hue='y',data=df)
df['age'].describe()
df_age_out = df[ df['age'] < 18 ] + df[df['age']>80]
df = df.drop(df_age_out.index, axis=0)

df['Age_band']=0
df.loc[df['age']<=27,'Age_band']= 'Young'
df.loc[(df['age']>27)&(df['age']<=45),'Age_band']= 'Adult'
df.loc[(df['age']>45)&(df['age']<=60),'Age_band']='Experienced'
df.loc[df['age']>60,'Age_band']= 'Senior'
df.head()

df.drop('age', axis=1,inplace=True)

####Job#####
df['job'].describe()
df_job_out = df[ df['job'] == 'unknown']
df = df.drop(df_job_out.index, axis=0)
df['job'] = df['job'].replace(['management', 'admin.'], 'white-collar')
df['job'] = df['job'].replace(['services','housemaid'], 'pink-collar')
df['job'] = df['job'].replace(['retired', 'student', 'unemployed'], 'other')

sns.countplot('job', hue='y', data=df)
#####marital####
df['marital'].describe()
df_m_out = df[ df['marital'] == 'unknown']
df = df.drop(df_m_out.index, axis=0)
sns.countplot('marital', hue='y', data=df)


####education####
df['education'].describe()
sns.countplot('education', hue='y', data=df)
df_ed_out = df[ df['education'] == 'illiterate']
df = df.drop(df_ed_out.index, axis=0)
df['education'] = df['education'].replace(['basic.4y', 'basic.6y', 'basic.9y'], 'school')


####loan######
df_loan_out = df[ df['loan'] == 'unknown']
df = df.drop(df_loan_out.index, axis=0)
df["loan_cat"] = df['loan'].map({'yes':1, 'no':0})
df.drop('loan', axis=1, inplace=True)
sns.countplot('loan_cat', hue='y', data=df)

####default#####
df_d_out = df[ df['default'] == 'yes']
df = df.drop(df_d_out.index, axis=0)
df['default_cat'] = df['default'].map( {'unknown':1, 'no':0} )
df.drop('default', axis=1,inplace = True)
df['default_cat'].value_counts()

####day_of_week####
sns.countplot('day_of_week', hue='y', data=df)
df.drop('day_of_week', axis=1, inplace=True)

###duration####
df.drop('duration', axis=1, inplace=True)

####campaign####
sns.countplot('campaign', hue='y', data=df)
df['campaign'].describe()
df_c_out = df[ df['campaign'] >12]
df = df.drop(df_c_out.index, axis=0)


###pdays###
sns.countplot('pdays', hue='y', data=df)
df.drop('pdays', axis=1, inplace=True)


###previous###
sns.countplot('previous', hue='y', data=df)
df['previous'].describe()
df_p = df[ df['previous'] > 4]
df =df.drop(df_p.index, axis=0)

####housing####
df_h_out = df[ df['housing'] == 'unknown']
df = df.drop(df_h_out.index, axis=0)
df["housing_cat"] = df['housing'].map({'yes':1, 'no':0})
df.drop('housing', axis=1, inplace=True)
sns.countplot('housing_cat', hue='y', data=df)

###poutcome###
df['poutcome'].describe()
sns.countplot('poutcome', hue='y', data=df)

#############################################################

one_hot = pd.get_dummies(df[['marital','education','job','Age_band','contact','month']])
df = df.join(one_hot)

df.drop(['marital','education','job','Age_band','contact','month'], axis=1,inplace = True)

one_hot1 = pd.get_dummies(df['poutcome'])
df = df.join(one_hot1)
df.drop('poutcome', axis=1,inplace = True)

df.drop('euribor3m', axis=1, inplace=True)
df.drop('nr.employed', axis=1, inplace=True)
df.drop('cons.price.idx', axis=1, inplace=True)
df.drop(['emp.var.rate','cons.conf.idx'], axis=1, inplace=True)

###############################################################

df['y_label'] = df['y'].map({'yes':1, 'no':0})
df.drop('y', axis=1, inplace=True)

###############################################################
from sklearn.utils import shuffle
df = shuffle(df)

X= df.iloc[:, :39]
y = df.iloc[:, 39]

from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier



clf1 = LogisticRegression(random_state=1)
clf2 = RandomForestClassifier(random_state=1)
clf5 =  KNeighborsClassifier(n_neighbors=5, metric = 'euclidean', weights = 'distance')


scores = cross_val_score(clf5, X, y, cv=10, scoring='accuracy')
print(scores.mean())


scores = cross_val_score(clf1, X, y, cv=10, scoring='accuracy')
print(scores.mean())

scores = cross_val_score(clf2, X, y, cv=10, scoring='accuracy')
print(scores.mean())


###################################################################