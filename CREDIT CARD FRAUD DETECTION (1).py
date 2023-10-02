#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# #IMPORTING DATASET

# In[2]:


Data = pd.read_csv('C:/CAREER/CREDIT.csv')
print(Data)


# # EXPLORATORY DATA ANALYSIS

# In[3]:


Data.head(10)


# In[4]:


Data.tail()


# In[5]:


Data.head()


# In[6]:


Data.shape


# In[7]:


Data.info()


# In[8]:


Data.describe()


# # DATA PREPROCESSING the following steps are executed in this step i) Finding missing values and dropping rows with missing values ii) Converting the data to Numpy iii) Dividing the data set into training data and test data

# In[9]:


Data.isnull().sum()


# In[10]:


Data.duplicated().sum()


# In[11]:


Data.loc[Data.duplicated() & Data['Class']==1]


# I will drop all duplicated transactions which might be errors during data collection

# In[12]:


Data.drop_duplicates(inplace=True)


# # DATA VISUALIZATION

# In[13]:


plt.figure(figsize = (12, 6))
sns.histplot(Data['Class'], bins = 20)


# In[14]:


import seaborn as sns
sns.countplot(Data.Class)


# In[15]:


from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.figure_factory as ff
class_0 = Data.loc[Data['Class'] == 0]["Time"]
class_1 = Data.loc[Data['Class'] == 1]["Time"]

hist_data = [class_0, class_1]
group_labels = ['Not Fraud', 'Fraud']

fig = ff.create_distplot(hist_data, group_labels, show_hist=False, show_rug=False)
fig['layout'].update(title='Credit Card Transactions Time Density Plot', xaxis=dict(title='Time [s]'))
iplot(fig, filename='dist_only')


# In[16]:


# The first way we can plot things is using the .plot extension from Pandas dataframes
# We'll use this to make a scatterplot of the Credit Card features V3 and V4.
Data.plot(kind="scatter", x="V3", y="V4")


# In[17]:


# The first way we can plot things is using the .plot extension from Pandas dataframes
# We'll use this to make a scatterplot of the Credit Card features V3 and V4.
Data.plot(kind="scatter", x="Time", y="V4")


# In[19]:


# We can also use the seaborn library to make a similar plot
# A seaborn jointplot shows bivariate scatterplots and univariate histograms in the same figure
sns.jointplot(x="V3", y="V4", data=Data, height=5)


# In[20]:


# One piece of information missing in the plots above is what species each plant is
# We'll use seaborn's FacetGrid to color the scatterplot by species
sns.FacetGrid(Data, hue="Class", size=5)    .map(plt.scatter, "V1", "V2")    .add_legend()


# In[21]:


sns.scatterplot(x='V1', y='V2', data=Data,
               hue='Class')
plt.show()


# In[22]:


plt.plot(Data['Time'])
plt.plot(Data['Class'])
 
# Adding Title to the Plot
plt.title("Scatter Plot")
 
# Setting the X and Y labels
plt.xlabel('Time')
plt.ylabel('Class')
 
plt.show


# In[23]:


from bokeh.plotting import figure, output_file, show
from bokeh.palettes import magma


# In[24]:


# instantiating the figure object
graph = figure(title = "Bokeh Scatter Graph")
color = magma(256)

# plotting the graph
graph.scatter(Data['Amount'], Data['V1'], color=color)
 
# displaying the model
show(graph)


# In[25]:


# plotting the graph
graph.vbar(Data['Amount'], top=Data['V2'],
           legend_label = "V3 VS V4", color='green')
 
graph.vbar(Data['V1'], top=Data['V1'],
           legend_label = "V2 VS V1", color='red')
 
graph.legend.click_policy = "hide"
 
# displaying the model
show(graph)


# In[26]:


import plotly.express as px
import pandas as pd

# plotting the scatter chart
fig = px.scatter(Data, x="V1", y="V2", color='Class')
 
# showing the plot
fig.show()


# In[27]:


# plotting the scatter chart
fig = px.histogram(Data, x='V10', color='Class')
 
# showing the plot
fig.show()


# In[28]:


# plotting the scatter chart
fig = px.line(Data, y='V4', color='Class')
 
# showing the plot
fig.show()


# In[29]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score,ConfusionMatrixDisplay,precision_score,recall_score,f1_score,roc_auc_score,roc_curve,precision_recall_curve
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import BaggingClassifier,RandomForestClassifier,VotingClassifier,AdaBoostClassifier


# In[30]:


x = Data.drop('Class', axis=1)
y = Data['Class']
#print(X_features.head(10))
#data.replace([np.inf, -np.inf], np.nan, inplace=True)
#X_train, X_test, target_train, target_test = train_test_split(X, target, test_size = 0.20)
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=102)


# In[31]:


print(f'percentage of Fraud in train set {len(y_train[y_train.values==1])/len(x_train)*100:.2f} %')
print(f'percentage of non-Fraud in train set {len(y_train[y_train.values==0])/len(x_train)*100:.2f} %')
print(f'percentage of Fraud in test set {len(y_val[y_val.values==1])/len(x_val)*100:.2f} %')
print(f'percentage of non-Fraud in test set {len(y_val[y_val.values==0])/len(x_val)*100:.2f} %')


# In[32]:


x.shape, y.shape


# In[33]:


print(x_train.shape)
print(x_val.shape)
print(y_train.shape)
print(y_val.shape)


# In[34]:


from sklearn.linear_model import LogisticRegression
logreg= LogisticRegression()
logreg.fit(x_train,y_train)
print(logreg)


# In[ ]:


#LogReg_clf = LogisticRegression()
DTree_clf = DecisionTreeClassifier()
RF_clf = RandomForestClassifier()

#LogReg_clf.fit(X_train, y_train)
DTree_clf.fit(x_train, y_train)
RF_clf.fit(x_train, y_train)

#LogReg_pred = LogReg_clf.predict(X_val)
DTree_pred = DTree_clf.predict(x_val)
RF_pred =RF_clf.predict(x_val)

#averaged_preds = (LogReg_pred + DTree_pred + RF_pred)//3
averaged_preds = (DTree_pred + RF_pred)//2
acc = accuracy_score(y_val, averaged_preds)
from sklearn.metrics import accuracy_score, f1_score, log_loss
l_loss = log_loss(y_val, averaged_preds)
f1 = f1_score(y_val, averaged_preds)

print("Accuracy is: " + str(acc))
print("Log Loss is: " + str(l_loss))
print("F1 Score is: " + str(f1))
print(classification_report(y_val,averaged_preds))
print(acc)


# In[ ]:


DTree_clf = DecisionTreeClassifier()
RF_clf = RandomForestClassifier()
voting_clf = VotingClassifier(estimators=[('DTree', DTree_clf), ('RF', RF_clf)], voting='hard')

voting_clf.fit(x_train, y_train)
preds = voting_clf.predict(x_val)
acc = accuracy_score(y_val, preds)
l_loss = log_loss(y_val, preds)
f1 = f1_score(y_val, preds)

print("Accuracy is: " + str(acc))
print("Log Loss is: " + str(l_loss))
print("F1 Score is: " + str(f1))


# In[ ]:


from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, ExtraTreesClassifier
#logreg_bagging_model = BaggingClassifier(base_estimator=LogReg_clf, n_estimators=50,)
dtree_bagging_model = BaggingClassifier(base_estimator=DTree_clf, n_estimators=50,)
random_forest = RandomForestClassifier(n_estimators=50,)
extra_trees = ExtraTreesClassifier(n_estimators=50,)

def bagging_ensemble(model):
    k_folds = KFold(n_splits=20,)
    results = cross_val_score(model, x_train, y_train, cv=k_folds)
    print(results.mean())

#bagging_ensemble(logreg_bagging_model)
bagging_ensemble(dtree_bagging_model)
bagging_ensemble(random_forest)
bagging_ensemble(extra_trees)


# In[ ]:


lg_pred_proba = lg.predict_proba(X_test)[:,1]
# calculate AUC of model
auc_lg = roc_auc_score(y_val, lg_pred_proba)
fpr_lg, tpr_lg, _ = roc_curve(y_val,  lg_pred_proba)
bag_pred_proba = bag_clf.predict_proba(x_val)[:,1]
# calculate AUC of model
auc_bag = roc_auc_score(y_test, bag_pred_proba)
fpr_bag, tpr_bag, _ = roc_curve(y_test,  bag_pred_proba)
vot_pred_proba = voting_clf.predict_proba(x_val)[:,1]
# calculate AUC of model
auc_vot = roc_auc_score(y_test, vot_pred_proba)
fpr_vot, tpr_vot, _ = roc_curve(y_val,  vot_pred_proba)
plt.plot(fpr_vot,tpr_vot,label="voting classifier, auc="+str(auc_vot))
plt.plot(fpr_lg,tpr_lg,label="logistic regression, auc="+str(auc_lg))
plt.plot(fpr_bag,tpr_bag,label="bagging classifier, auc="+str(auc_bag))
plt.legend(loc=4)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate or recall')
plt.show()


# In[ ]:


matrix = confusion_matrix(y_val, vot_pred)
matrix_display = ConfusionMatrixDisplay(matrix,display_labels=['Not Fraud','Fraud'])
matrix_display.plot()
plt.title('Confusion matrix for Voting classifier')
plt.show()


# In[ ]:





# In[ ]:




