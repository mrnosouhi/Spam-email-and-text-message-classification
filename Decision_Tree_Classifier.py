
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.svm import SVC
import itertools
from sklearn.metrics import confusion_matrix




def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    
    #This function prints and plots the confusion matrix.
    
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0)
    plt.yticks(tick_marks, classes)



    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


class_names = [0,1]

# Reading data from the fraud_payment CSV file 
df = pd.read_csv('./spambase/spambase.data')

X = df.iloc[:,:57]
print(X.shape)
columns = ['word_freq_make','word_freq_address','word_freq_all','word_freq_3d','word_freq_our','word_freq_over','word_freq_remove','word_freq_internet','word_freq_order','word_freq_mail','word_freq_receive','word_freq_will','word_freq_people','word_freq_report','word_freq_addresses','word_freq_free','word_freq_business','word_freq_email','word_freq_you','word_freq_credit','word_freq_your','word_freq_font','word_freq_000','word_freq_money','word_freq_hp','word_freq_hpl','word_freq_george','word_freq_650','word_freq_lab','word_freq_labs','word_freq_telnet','word_freq_857','word_freq_data','word_freq_415','word_freq_85','word_freq_technology','word_freq_1999','word_freq_parts','word_freq_pm','word_freq_direct','word_freq_cs','word_freq_meeting','word_freq_original','word_freq_project','word_freq_re','word_freq_edu','word_freq_table','word_freq_conference','char_freq_;','char_freq_(','char_freq_[','char_freq_!','char_freq_$','char_freq_#','capital_run_length_average','capital_run_length_longest','capital_run_length_total']
X.columns = columns
y = df.iloc[:,57]

# Split the dataset 
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.33,random_state=0)


# Decision Tree
tree = DecisionTreeClassifier(random_state=0)
tree.fit(X_train, y_train)


# How many correct predictions? Using a confusion matrix we can see how many correct prediction the modal has
y_pred = tree.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)
    
#Plotting confusion matrix based on the real labels and the predictions on the test set 

plt.figure()
plot_confusion_matrix(cm, class_names,
                          normalize=False, title='Decision Tree Confusion matrix for the default max_depth', cmap=plt.cm.Blues)
plt.show()


## Plotting feature importances for Decision Tree 
fig, ax  = plt.subplots(figsize=(7, 12), dpi=100)
ax.barh(columns, tree.feature_importances_)
ax.set_xlabel("Importance of Feature")
ax.set_ylabel("Feature")
ax.set_title("Decision Tree Feature Importance without max_depth")



# Changing max_depth for the DecisionTree
tree4 = DecisionTreeClassifier(max_depth=4, random_state=0)
tree4.fit(X_train, y_train)


# How many correct predictions? Using a confusion matrix we can see how many correct prediction the modal has
y_pred = tree4.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)
    
#Plotting confusion matrix based on the real labels and the predictions on the test set 

plt.figure()
plot_confusion_matrix(cm, class_names,
                          normalize=False, title='Decision Tree Confusion matrix for max_depth = 4', cmap=plt.cm.Blues)
plt.show()

## Plotting feature importances for Decision Tree 
fig, ax  = plt.subplots(figsize=(7, 12), dpi=100)
ax.barh(columns, tree4.feature_importances_)
ax.set_xlabel("Importance of Feature")
ax.set_ylabel("Feature")
ax.set_title("Decision Tree Feature Importance with max_depth=4")


