#!/usr/bin/env python
# coding: utf-8

# In[135]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, auc, confusion_matrix, precision_score, recall_score, roc_curve, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.tree import export_graphviz
from IPython.display import Image
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import BaggingClassifier # for bagging regression problem
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing


# In[136]:


Org_data = pd.read_csv('weatherAUS.csv')
df =pd.read_csv('weatherAUS.csv')


# In[137]:


df.head()


# In[138]:


df.describe()


# In[139]:


df.tail() 


# In[140]:


df.shape


# In[141]:


df.drop_duplicates() 
df.shape


# In[142]:


df.columns


# In[143]:


df.info()


# In[144]:


df['Date']=pd.to_datetime(df['Date'])


# In[145]:


df['Year'] = df['Date'].dt.year
df['Year'].head()


# In[146]:


df['Month'] = df['Date'].dt.month
df['Month'].head()


# In[147]:


df['Day'] = df['Date'].dt.day
df['Day'].head()


# In[148]:


df.drop('Date', axis=1, inplace = True)
df.info()


# In[149]:


categorical = [catvar for catvar in df.columns if df[catvar].dtype=='O']
print('There are {} categorical variables\n'.format(len(categorical)))
print('The categorical variables are :', categorical)


# In[150]:


numerical = [var for var in df.columns if df[var].dtype!='O']
print('There are {} numerical variables\n'.format(len(numerical)))
print('The numerical variables are :', numerical)


# In[151]:


missing_values=df.isnull().sum() # missing values.
percent_miss_value = df.isnull().sum()/df.shape[0]*100 # missing value %
miss_value_info = {
    'missing_values ':missing_values,
    'percent_missing %':percent_miss_value , 
     'data type' : df.dtypes
}
miss_value_df=pd.DataFrame(miss_value_info)
miss_value_df


# In[152]:


df.nunique()


# In[153]:


df['RainToday'].replace({'No': 0, 'Yes': 1},inplace = True)
df['RainTomorrow'].replace({'No': 0, 'Yes': 1},inplace = True)


# In[154]:


from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer


# In[155]:


df.reset_index(drop=True, inplace=True)
my_imputer = IterativeImputer()
df_numerical = df._get_numeric_data()
df_numerical_columns = df_numerical.columns 
print(df_numerical.shape)
df_numerical.isnull().sum()


# In[156]:


df_imputed = my_imputer.fit_transform(df_numerical)
df_imputed = pd.DataFrame(df_imputed, columns=df_numerical_columns)
df_imputed.isnull().sum()


# In[157]:


df['RainToday']=df['RainToday'].fillna(df['RainToday'].mode()[0])
df['RainTomorrow']=df['RainTomorrow'].fillna(df['RainTomorrow'].mode()[0])
df['WindDir9am'] = df['WindDir9am'].fillna(df['WindDir9am'].mode()[0])
df['WindGustDir'] = df['WindGustDir'].fillna(df['WindGustDir'].mode()[0])
df['WindDir3pm'] = df['WindDir3pm'].fillna(df['WindDir3pm'].mode()[0])
df['Location']=df['Location'].fillna(df['Location'].mode()[0])


# In[158]:


missing_values=df.isnull().sum() # missing values.
percent_miss_value = df.isnull().sum()/df.shape[0]*100 # missing value %
miss_value_info = {
    'missing_values ':missing_values,
    'percent_missing %':percent_miss_value , 
     'data type' : df.dtypes
}
miss_value_df=pd.DataFrame(miss_value_info)
miss_value_df


# In[159]:


df['MinTemp']=df_imputed['MinTemp']
df['MaxTemp']=df_imputed['MaxTemp']
df['Rainfall']=df_imputed['Rainfall']
df['Evaporation']=df_imputed['Evaporation']
df['Sunshine']=df_imputed['Sunshine']
df['WindGustSpeed']=df_imputed['WindGustSpeed']
df['WindSpeed9am']=df_imputed['WindSpeed9am']
df['WindSpeed3pm']=df_imputed['WindSpeed3pm']
df['Humidity9am']=df_imputed['Humidity9am']
df['Humidity3pm']=df_imputed['Humidity3pm']
df['Pressure9am']=df_imputed['Pressure9am']
df['Pressure3pm']=df_imputed['Pressure3pm']
df['Cloud9am']=df_imputed['Cloud9am']
df['Cloud3pm']=df_imputed['Cloud3pm']
df['Temp9am']=df_imputed['Temp9am']
df['Temp3pm']=df_imputed['Temp3pm']


# In[160]:


le = preprocessing.LabelEncoder()
df['Location'] = le.fit_transform(df['Location'])
df['WindDir9am'] = le.fit_transform(df['WindDir9am'])
df['WindDir3pm'] = le.fit_transform(df['WindDir3pm'])
df['WindGustDir'] = le.fit_transform(df['WindGustDir'])


# In[161]:


df.isnull().sum()


# In[162]:


df[numerical].isnull().sum()
df[numerical].describe()
ax = sns.boxplot(df["Rainfall"], orient="h", palette="Set2")


# In[163]:


plt.figure(figsize=(12,10))
sns.heatmap(df.corr(),annot=True)


# In[164]:


df=df.drop(['Temp3pm','Temp9am','Humidity9am'],axis=1)


# In[165]:


plt.figure(figsize=(15,10))

plt.subplot(2, 2, 1)
fig = df.boxplot(column='Rainfall')
fig.set_title('')
fig.set_ylabel('Rainfall')

plt.subplot(2, 2, 2)
fig = df.boxplot(column='MinTemp')
fig.set_title('')
fig.set_ylabel('Mintemp')

plt.subplot(2, 2, 3)
fig = df.boxplot(column='MaxTemp')
fig.set_title('')
fig.set_ylabel('Maxtemp')

plt.subplot(2, 2, 4)
fig = df.boxplot(column='Evaporation')
fig.set_title('')
fig.set_ylabel('Evaporation')


# In[166]:


plt.figure(figsize=(15,10))
plt.subplot(2, 2, 1)
fig = df.boxplot(column='WindSpeed9am')
fig.set_title('')
fig.set_ylabel('WindSpeed9am')

plt.subplot(2, 2, 2)
fig = df.boxplot(column='WindSpeed3pm')
fig.set_title('')
fig.set_ylabel('WindSpeed3pm')


# In[167]:


plt.figure(figsize=(15,10))

plt.subplot(2, 2, 1)
fig = df.Rainfall.hist(bins=10)
fig.set_xlabel('Rainfall')
fig.set_ylabel('RainTomorrow')

plt.subplot(2, 2, 2)
fig = df.Evaporation.hist(bins=10)
fig.set_xlabel('Evaporation')
fig.set_ylabel('RainTomorrow')

plt.subplot(2, 2, 3)
fig = df.WindSpeed9am.hist(bins=10)
fig.set_xlabel('WindSpeed9am')
fig.set_ylabel('RainTomorrow')

plt.subplot(2, 2, 4)
fig = df.WindSpeed3pm.hist(bins=10)
fig.set_xlabel('WindSpeed3pm')
fig.set_ylabel('RainTomorrow')


# In[168]:


IQR = df.Rainfall.quantile(0.75) - df.Rainfall.quantile(0.25)
Lower_fence = df.Rainfall.quantile(0.25) - (IQR * 3)
Upper_fence = df.Rainfall.quantile(0.75) + (IQR * 3)
df.describe()
print('Rainfall outliers are values < {lowerboundary} or > {upperboundary}'.format(lowerboundary=Lower_fence, upperboundary=Upper_fence))


# In[169]:


df.drop(df[df['Rainfall'] < Lower_fence].index, inplace = True)
df.drop(df[df['Rainfall'] > Upper_fence].index, inplace = True)


# In[170]:


plt.subplot(2, 2, 1)
fig = df.boxplot(column='Rainfall')
fig.set_title('')
fig.set_ylabel('Rainfall')
df.describe()


# In[171]:


IQR = df.Evaporation.quantile(0.75) - df.Evaporation.quantile(0.25)
Lower_fence = df.Evaporation.quantile(0.25) - (IQR * 3)
Upper_fence = df.Evaporation.quantile(0.75) + (IQR * 3)
print('Evaporation outliers are values < {lowerboundary} or > {upperboundary}'.format(lowerboundary=Lower_fence, upperboundary=Upper_fence))
df.describe()


# In[172]:


df.drop(df[df['Evaporation']< Lower_fence].index, inplace = True)
df.drop(df[df['Evaporation'] > Upper_fence].index, inplace = True)

plt.subplot(2, 2, 1)
fig = df.boxplot(column='Evaporation')
fig.set_title('')
fig.set_ylabel('Evaporation')
df.describe()


# In[173]:


IQR = df.WindSpeed9am.quantile(0.75) - df.WindSpeed9am.quantile(0.25)
Lower_fence = df.WindSpeed9am.quantile(0.25) - (IQR * 3)
Upper_fence = df.WindSpeed9am.quantile(0.75) + (IQR * 3)
print('WindSpeed9am outliers are values < {lowerboundary} or > {upperboundary}'.format(lowerboundary=Lower_fence, upperboundary=Upper_fence))


# In[174]:


df.drop(df[df['WindSpeed9am']< Lower_fence].index, inplace = True)
df.drop(df[df['WindSpeed9am'] > Upper_fence].index, inplace = True)
plt.subplot(2, 2, 1)
fig = df.boxplot(column='WindSpeed9am')
fig.set_title('')
fig.set_ylabel('WindSpeed9am')
df.describe()


# In[175]:


IQR = df.WindSpeed3pm.quantile(0.75) - df.WindSpeed3pm.quantile(0.25)
Lower_fence = df.WindSpeed3pm.quantile(0.25) - (IQR * 3)
Upper_fence = df.WindSpeed3pm.quantile(0.75) + (IQR * 3)
print('WindSpeed3pm outliers are values < {lowerboundary} or > {upperboundary}'.format(lowerboundary=Lower_fence, upperboundary=Upper_fence))


# In[176]:


df.drop(df[df['WindSpeed3pm']< Lower_fence].index, inplace = True)
df.drop(df[df['WindSpeed3pm'] > Upper_fence].index, inplace = True)
plt.subplot(2, 2, 1)
fig = df.boxplot(column='WindSpeed3pm')
fig.set_title('')
fig.set_ylabel('WindSpeed3pm')
df.describe()


# In[177]:


df.describe()


# In[178]:


X = df.drop('RainTomorrow',axis=1)
X


# In[179]:


y = df['RainTomorrow']
y


# In[180]:


features_label = X.columns


# In[181]:


from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 50, criterion = 'entropy', random_state = 0)
classifier.fit(X, y)
importances = classifier.feature_importances_
print(importances)


# In[182]:


indices = np.argsort(importances)[::-1]
for i in range(X.shape[1]):
    print ("%2d) %-*s %f" % (i + 1, 30, features_label[i],importances[indices[i]]))


# In[183]:


plt.title('Feature Importances')
plt.bar(range(X.shape[1]),importances[indices], color="green", align="center")
plt.xticks(range(X.shape[1]),features_label, rotation=90)
plt.xlim([-1, X.shape[1]])
plt.show()


# In[184]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection  import train_test_split
from sklearn.metrics import confusion_matrix,accuracy_score,auc


# In[185]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 42)


# In[186]:


Random = RandomForestClassifier(n_estimators=10)
Random.fit(X_train, y_train)


# In[187]:


y_pred = Random.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test,y_pred)


# In[188]:


cm


# In[189]:


accuracy


# In[190]:


X_filter = X[['Location', 'MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation', 'Sunshine',
             'WindGustDir', 'WindGustSpeed', 'WindDir9am']]


# In[191]:


X_train, X_test, y_train, y_test = train_test_split(X_filter, y, test_size = 0.20, random_state = 42, stratify=y)


# In[192]:


Random = RandomForestClassifier(n_estimators=10)
Random.fit(X_train, y_train)


# In[193]:


y_pred = Random.predict(X_test) 
cm = confusion_matrix(y_test, y_pred)
acc = accuracy_score(y_test, y_pred)


# In[194]:


cm


# In[195]:


accuracy


# In[196]:


X_filter = X[['Location', 'MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation', 'Sunshine',
             'WindGustDir', 'WindGustSpeed', 'WindDir9am']]


# In[197]:


X_train, X_test, y_train, y_test = train_test_split(X_filter, y, test_size = 0.20, random_state = 42, stratify=y)


# In[198]:


Random = RandomForestClassifier(n_estimators=10)

Random.fit(X_train, y_train)


# In[199]:


y_pred = Random.predict(X_test) 
cm = confusion_matrix(y_test, y_pred)
acc = accuracy_score(y_test, y_pred)


# In[200]:


cm


# In[201]:


acc


# In[202]:


y_pred_rf_test=Random.predict_proba(X_test)
y_pred_rf_test=y_pred_rf_test[:,1]
y_pred_rf_test


# In[203]:


fpr_rf_test, tpr_rf_test, th_rf_test = roc_curve(y_test, y_pred_rf_test)


# In[204]:


from sklearn.model_selection import RandomizedSearchCV


# In[205]:


n_estimators = [10,20,30,40,50,60]
min_samples_split = [5, 10, 15, 20, 25, 30]
random_grid = {'n_estimators': n_estimators,
               'min_samples_split': min_samples_split}


# In[206]:


rf = RandomForestClassifier()
rf_random = RandomizedSearchCV(estimator=rf, param_distributions=random_grid, cv=5, random_state=42)
rf_random.fit(X_train, y_train)


# In[207]:


rf_random.best_params_


# In[208]:


Random = RandomForestClassifier(n_estimators=60, min_samples_split=15, max_features='sqrt', 
                               bootstrap=True, oob_score=True, random_state=42)

Random.fit(X_train, y_train)


# In[ ]:


y_pred = Random.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
acc = accuracy_score(y_test, y_pred)


# In[209]:


cm


# In[210]:


acc


# In[211]:


from sklearn.linear_model import LogisticRegression
Logistic = LogisticRegression(class_weight='balanced')
Logistic.fit(X_train, y_train)


# In[212]:


y_pred = Logistic.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
acc = accuracy_score(y_test, y_pred)


# In[213]:


cm


# In[214]:


acc


# In[215]:


y_pred_lr_test=Logistic.predict_proba(X_test)
y_pred_lr_test=y_pred_lr_test[:,1]
y_pred_lr_test


# In[216]:


fpr_lr_test, tpr_lr_test, th_lr_test = roc_curve(y_test, y_pred_lr_test)


# In[217]:


from sklearn.neighbors import KNeighborsClassifier
KNN = KNeighborsClassifier()
KNN.fit(X_train, y_train)


# In[218]:


y_pred = KNN.predict(X_test)

# Making the Confusion Matrix 
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
acc = accuracy_score(y_test, y_pred)


# In[219]:


cm


# In[220]:


acc


# In[221]:


y_pred_knn_test=KNN.predict_proba(X_test)
y_pred_knn_test=y_pred_knn_test[:,1]
y_pred_knn_test


# In[222]:


fpr_knn_test, tpr_knn_test, th_knn_test = roc_curve(y_test, y_pred_knn_test)


# In[223]:


from sklearn.ensemble import GradientBoostingClassifier
GradientBoo = GradientBoostingClassifier()
GradientBoo.fit(X_train, y_train)


# In[224]:


y_pred = GradientBoo.predict(X_test)

# Making the Confusion Matrix 
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
acc = accuracy_score(y_test, y_pred)
cm


# In[225]:


acc


# In[226]:


y_pred_gb_test=GradientBoo.predict_proba(X_test)
y_pred_gb_test=y_pred_gb_test[:,1]
y_pred_gb_test


# In[227]:


fpr_gb_test, tpr_gb_test, th_gb_test = roc_curve(y_test, y_pred_gb_test)


# In[228]:


from sklearn.preprocessing import StandardScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.pipeline import Pipeline


# In[229]:


pipeline = Pipeline([
('imputer', IterativeImputer()),
('scaler', StandardScaler())]) 


# In[230]:


X_transformed = pipeline.fit_transform(X)
X_transformed


# In[231]:


X_train,X_test_nb,y_train,y_test_nb= train_test_split(X_transformed,y,test_size=0.2,stratify=y,random_state = 51)


# In[232]:


X_train.shape,y_train.shape, X_test_nb.shape,y_test_nb.shape


# In[233]:


model=GaussianNB()


# In[234]:


model.fit(X_train, y_train)


# In[235]:


model.score(X_train,y_train)


# In[236]:


pred_y=model.predict(X_test_nb)
pred_y


# In[237]:


confusion_matrix(y_test_nb, pred_y)


# In[238]:


accuracy_score(y_test_nb, pred_y)


# In[239]:


y_pred_naive_test=model.predict_proba(X_test_nb)
y_pred_naive_test=y_pred_naive_test[:,1]
y_pred_naive_test


# In[240]:


fpr_naive_test, tpr_naive_test, th_naive_test = roc_curve(y_test_nb, y_pred_naive_test)


# In[241]:


plt.plot(fpr_rf_test, tpr_rf_test, label = "Random Forest on the Test Set")
plt.plot(fpr_lr_test, tpr_lr_test, label = "Logistic Regression on the Test Set")
plt.plot(fpr_knn_test, tpr_knn_test, label = "KNN on the Test Set")
plt.plot(fpr_gb_test, tpr_gb_test, label = "Gradient Boosting on the Test Set")
plt.plot(fpr_naive_test, tpr_naive_test, label = "Naive Bayes on the Test Set")
plt.plot([0, 1], [0, 1], 'k:')
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.title('ROC Curve for Rainfall Prediction')
plt.legend()
plt.show()


# In[242]:


import pickle


# In[243]:


pickle.dump(Random, open('model_final.pkl','wb'))


# In[244]:


X_test.head()


# In[245]:


X_test.columns


# In[246]:


y_test.head()


# In[247]:


model = pickle.load(open('model_final.pkl','rb'))
print(model)


# In[248]:


print(model.predict([[36,9.4,18.1,1.4,2.914436,6.998996,13,43.0,14]]))


# In[250]:


import flask
app = flask(__name__)

app@route("/")

def hello():
    return "hello world"

if __name__ == "__main__":
    app.run()

