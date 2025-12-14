#!/usr/bin/env python
# coding: utf-8

# # Heart Disease Classification - Machine Learning Project
# LUCAS Antoine
# BESOMBES Gabriel
# March 2019
# #

# ## Introduction
# ---
# We are analyzing heart disease data. The goal is to predict whether a person
# has heart disease based on various health indicators.
# We will explore the dataset and test different classifiers from sklearn.
#
# ---
# #

# ## Imports
# ---
# We will use the following libraries:
# * pandas for loading and viewing CSV data
# * matplotlib for visualization
# * sklearn for classifiers and evaluation metrics

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn import metrics


# ---
# #

# ## Loading the Data
# ---
# Load the _heart.csv_ file located in the project root using *pandas.read_csv()*
# Dataset available at: https://www.kaggle.com/ronitf/heart-disease-uci

# In[2]:


heart = pd.read_csv("heart.csv")


# Preview the first and last rows of the dataframe

# In[3]:


heart.head()


# In[4]:


heart.tail()


# ---
# #

# ## Data Overview
# ---
# Use _describe_ to get a quick statistical summary

# In[5]:


heart.describe()


# Check for missing values:
# * _isnull()_ identifies missing values
# * _any(axis=1)_ returns True for rows with any missing value
#
# If there were missing values, we would see True in the output

# In[6]:


for i in heart.isnull().any(axis=1):
    if i:
        print(i)


# ---
# #

# ## K-Nearest Neighbors (KNN)
# ---
# Test with a KNN classifier using k=3

# In[7]:


knn = KNeighborsClassifier(n_neighbors=3)


# ###
# Default parameters for the KNN classifier:

# In[8]:


print(knn)


# ###
# Examine columns to identify the target variable

# In[9]:


heart.columns


# ###
# The target column contains binary values (1 = heart disease, 0 = no heart disease)

# In[10]:


heart["target"].head()


# ###
# We use *%%time* to measure execution time.

# ###
# Train-test split (70/30)

# In[11]:


get_ipython().run_cell_magic(
    "time",
    "",
    'X_train, X_test, y_train, y_test = train_test_split(heart.drop("target", axis=1),\n                                                    heart["target"],\n                                                    test_size=0.3)',
)


# ###
# Train the classifier

# In[12]:


get_ipython().run_cell_magic("time", "", "knn.fit(X_train, y_train)")


# ###
# Make predictions on test data

# In[13]:


get_ipython().run_cell_magic("time", "", "res=knn.predict(X_test)")


# ###
# Calculate accuracy

# In[14]:


get_ipython().run_cell_magic("time", "", "metrics.accuracy_score(y_test, res)")


# ###
# Confusion matrix

# In[15]:


cm = metrics.confusion_matrix(y_test, res)
print(cm)


# ###
# Visualize with pyplot

# In[16]:


fig, ax = plt.subplots()
im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
ax.figure.colorbar(im, ax=ax)

ax.set(
    xticks=np.arange(cm.shape[1]),
    yticks=np.arange(cm.shape[0]),
    xticklabels=["Diseased", "Healthy"],
    yticklabels=["Diseased", "Healthy"],
    title="Confusion Matrix",
    ylabel="Actual",
    xlabel="Predicted",
)


plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")


thresh = cm.max() / 2.0
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(
            j,
            i,
            format(cm[i, j], "d"),
            ha="center",
            va="center",
            color="white" if cm[i, j] > thresh else "black",
        )
fig.tight_layout()


# Most errors are false positives (healthy patients predicted as diseased):
# For healthy patients:
# * 26 correct predictions and 14 incorrect, error rate of 14/40*100=35%
#
# For diseased patients:
# * 47 correct predictions and 4 incorrect, error rate of 4/51*100=8%
#
# This type of error is preferable in medical contexts - better to have false positives
# (extra verification) than false negatives (missed diagnoses)

# ###
# Run multiple iterations with different k values to find the optimal k

# In[17]:


l = []
knns = []
data = heart.drop("target", axis=1)
target = heart["target"]
for n in range(1, 51):
    l.append([])
    knns.append([])
    for i in range(0, 100):
        knn = KNeighborsClassifier(n_neighbors=n)
        X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.3)
        knn.fit(X_train, y_train)
        knns[-1].append([knn, y_test, knn.predict(X_test)])
        l[-1].append(metrics.accuracy_score(y_test, knn.predict(X_test)))


# In[18]:


l2 = [sum(X) / len(X) for X in l]


# In[19]:


plt.plot(l2)


# ###
# Find the k value with the highest mean accuracy

# In[20]:


print(max(l2))
print(l2.index(max(l2)))


# The optimal k is around 19

# ###
# Find the single best classifier across all iterations

# In[21]:


print(max(max(l)))
print(max(l).index(max(max(l))))


# Best single model uses k=1

# In[22]:


print(l[0].index(max(l[0])))


# ###
# Retrieve this classifier and compute its confusion matrix

# In[23]:


knn, y_test, res = knns[0][8]


# In[24]:


cm = metrics.confusion_matrix(y_test, res)
print(cm)


# ###
# Visualize with pyplot

# In[25]:


fig, ax = plt.subplots()
im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
ax.figure.colorbar(im, ax=ax)

ax.set(
    xticks=np.arange(cm.shape[1]),
    yticks=np.arange(cm.shape[0]),
    xticklabels=["Diseased", "Healthy"],
    yticklabels=["Diseased", "Healthy"],
    title="Confusion Matrix",
    ylabel="Actual",
    xlabel="Predicted",
)


plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")


thresh = cm.max() / 2.0
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(
            j,
            i,
            format(cm[i, j], "d"),
            ha="center",
            va="center",
            color="white" if cm[i, j] > thresh else "black",
        )
fig.tight_layout()


# Similar error pattern but with fewer mistakes, matching the higher accuracy.

# ---
# #

# ## Intermediate Reflection
# ---
# KNN may not be the best approach for this dataset with many features.
# Let's test other classifiers.
#
# ---
# #

# ## Logistic Regression
# ---
# Test with logistic regression

# In[26]:


logreg = LogisticRegression(
    random_state=0, solver="lbfgs", multi_class="multinomial", max_iter=5000
)


# ###
# Logistic Regression parameters:

# In[27]:


print(logreg)


# ###
# Train-test split (70/30)

# In[28]:


get_ipython().run_cell_magic(
    "time",
    "",
    'X_train, X_test, y_train, y_test = train_test_split(heart.drop("target", axis=1),\n                                                    heart["target"],\n                                                    test_size=0.3)',
)


# ###
# Train the classifier

# In[29]:


get_ipython().run_cell_magic("time", "", "logreg.fit(X_train, y_train)")


# ###
# Make predictions on test data

# In[30]:


get_ipython().run_cell_magic("time", "", "res=logreg.predict(X_test)")


# ###
# Calculate accuracy

# In[31]:


get_ipython().run_cell_magic("time", "", "metrics.accuracy_score(y_test, res)")


# ###
# Run multiple iterations to find the best model

# In[32]:


l = []
logregs = []
data = heart.drop("target", axis=1)
target = heart["target"]
for i in range(0, 100):
    logreg = LogisticRegression(
        random_state=0, solver="lbfgs", multi_class="multinomial", max_iter=5000
    )
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.3)
    logreg.fit(X_train, y_train)
    logregs.append([logreg, y_test, logreg.predict(X_test)])
    l.append(metrics.accuracy_score(y_test, logreg.predict(X_test)))


# In[33]:


plt.plot(l)


# In[34]:


sum(l) / len(l)


# Over 100 iterations, some models achieve very high accuracy with a mean of ~82%.
# This classifier appears to perform better than KNN.

# ###
# Find the best classifier

# In[35]:


print(max(l))
print(l.index(max(l)))


# ###
# Retrieve this classifier and compute its confusion matrix

# In[36]:


logreg, y_test, res = logregs[84]


# In[37]:


cm = metrics.confusion_matrix(y_test, res)
print(cm)


# ###
# Visualize with pyplot

# In[38]:


fig, ax = plt.subplots()
im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
ax.figure.colorbar(im, ax=ax)

ax.set(
    xticks=np.arange(cm.shape[1]),
    yticks=np.arange(cm.shape[0]),
    xticklabels=["Diseased", "Healthy"],
    yticklabels=["Diseased", "Healthy"],
    title="Confusion Matrix",
    ylabel="Actual",
    xlabel="Predicted",
)


plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")


thresh = cm.max() / 2.0
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(
            j,
            i,
            format(cm[i, j], "d"),
            ha="center",
            va="center",
            color="white" if cm[i, j] > thresh else "black",
        )
fig.tight_layout()


# Even fewer errors, with more false positives than false negatives

# ---
# #

# ## Gaussian Naive Bayes
# ---

# In[39]:


GNB = GaussianNB()


# In[40]:


get_ipython().run_cell_magic(
    "time",
    "",
    'X_train, X_test, y_train, y_test = train_test_split(heart.drop("target", axis=1),\n                                                    heart["target"],\n                                                    test_size=0.3)',
)


# ###
# Train the classifier

# In[41]:


get_ipython().run_cell_magic("time", "", "GNB.fit(X_train, y_train)")


# ###
# Make predictions on test data

# In[42]:


get_ipython().run_cell_magic("time", "", "res=GNB.predict(X_test)")


# ###
# Calculate accuracy

# In[43]:


get_ipython().run_cell_magic("time", "", "metrics.accuracy_score(y_test, res)")


# Accuracy appears lower, but execution time is very fast

# ---
# #

# ## Support Vector Machine (SVM)
# ---

# In[44]:


svc = SVC(kernel="linear")


# In[45]:


get_ipython().run_cell_magic(
    "time",
    "",
    'X_train, X_test, y_train, y_test = train_test_split(heart.drop("target", axis=1),\n                                                    heart["target"],\n                                                    test_size=0.3)',
)


# ###
# Train the classifier

# In[46]:


get_ipython().run_cell_magic("time", "", "svc.fit(X_train, y_train)")


# ###
# Make predictions on test data

# In[47]:


get_ipython().run_cell_magic("time", "", "res=svc.predict(X_test)")


# ###
# Calculate accuracy

# In[48]:


get_ipython().run_cell_magic("time", "", "metrics.accuracy_score(y_test, res)")


# High accuracy again, but with longer execution time

# ---
# #

# ## Conclusion
# ---
# Models requiring more computational resources tend to be more accurate.
# Different classifiers are suited for different types of datasets.
# With more data, a trade-off between execution time and performance would be needed.
# In this case, simpler linear classifiers gave better results.
#
# ---
# #
