from cProfile import label
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import roc_curve
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.semi_supervised import LabelPropagation

def DataPrepr(data,features,train,enc = []):

    print(data)
    print("Feature Unique Value Analysis:")
    # for i in features:
    #     print(data[i].value_counts())

    #discarding columns of little to no need
    discard_columns = ["income","fnlwgt","education","capital gain", "capital loss"]



    #fixing the data to remove leading and ending spaces from each column
    #eg fixing " Married" to "Married"
    for i in features:
        if type(data[i].iloc[0]) == str:
            data[i] = data[i].str.strip()
    #removing all lines that have missing values in them (only a few were identified so we are ok to remove them)
        data = data.drop(data.index[data[i] == "?"])

    #keep the class as a separate DF to return it as "y_train/test"
    predict_class = data["income"]
    predict_class.replace("<=50K", 0, inplace=True)
    predict_class.replace(">50K", 1, inplace=True)

    #remove the unneeded columns from the column list and re-apply the columns in the DF
    for i in discard_columns:
        features.remove(i)

    data=data[features]



    #group all the countries under "Other" to remove outliers
    #every country will be either US, Mexico or Other
    #Note: all "?" countries have been already removed previously
    #from Train dataset analysis:
    #89% of values from USA, 2% Mexico, 1,7% missing values, 
    data['native country'] = np.where(data['native country'].isin(["United-States","Mexico"]), data['native country'], "other")


    #one-hot encoding method:
    #data = pd.get_dummies(data,columns=['workclass','marital_status','Occupation','relationship','race','sex','native country'],prefix=['workclass','marital_status','Occupation','relationship','race','sex','native country'])



    # create and store an encoder for each categorical variable in the dataset
    # attention: the LabelEncoder() encodes the categorical data based on order found 
    # e.g. if two values (Male, Female) are found in the dataset, IF Male is found first then male = 0 Female = 1
    # IF Female is found first then Female = 0 Male = 1
    # we need to store the encoding results from the training dataset and apply the exact same encoding to the test and unlabeled dataset.
    # the encoder results list (enc[]) is passed outside the def to be used elsewhere too.
    encoder = LabelEncoder()

    if train: #check if this is the training dataset or not
        enc = [] #create a list to store the results/order of the encoder
        for j,i in enumerate(features): #for each column in the dataset
            if type(data[i].iloc[0]) == str: #check if it contains categorical data (otherwise it would be a number)
                encoder.fit(data[[i]]) #encode categorical values to numbers *ascending order based on which is encountered first in the dataset
                enc.append(encoder.classes_) #append the encoding results in the list
                data[i] = encoder.transform(data[[i]]) #replace the str with the encoded values
            else:
                enc.append([]) #if the column does not contain categorical data append an empty list to keep up with the indexing
    else: #if this is the test dataset
        for j,i in enumerate(features): #for each column
            if type(data[i].iloc[0]) == str: #check if column has categorical variables
                encoder.classes_ = enc[j] #load the encoding from the training dataset to the current encoder
                data[i] = encoder.transform(data[[i]]) #replace the str with the encoded values


       


    print("Preprocessed Dataset:\n")
    print(data)



    return data, predict_class, enc





featurelist = ['age','workclass','fnlwgt', 'education', 'education_num', 'marital_status', 'Occupation', 'relationship', 'race', 'sex', 'capital gain', 'capital loss', 'hours per week', 'native country', 'income']
df1 = pd.read_csv("train-project.data", names=featurelist)
x_train, y_train, enc = DataPrepr(df1,featurelist,True)


featurelist = ['age','workclass','fnlwgt', 'education', 'education_num', 'marital_status', 'Occupation', 'relationship', 'race', 'sex', 'capital gain', 'capital loss', 'hours per week', 'native country', 'income']
df2 = pd.read_csv("test-project.data", names=featurelist)
x_test, y_test, enc = DataPrepr(df2,featurelist,False,enc)

featurelist = ['age','workclass','fnlwgt', 'education', 'education_num', 'marital_status', 'Occupation', 'relationship', 'race', 'sex', 'capital gain', 'capital loss', 'hours per week']
df3 = pd.read_csv('unlabeled-project.data', index_col=0,header=None)

df3.columns = featurelist
df3['native country'] = np.NaN
df3['income'] = -1
featurelist.extend(['native country','income'])
x_test_unlab, y_test_unlab, enc = DataPrepr(df3,featurelist,False,enc)

# summarize training set size
print('Labeled Train Set:', x_train.shape, y_train.shape)
print('Unlabeled Train Set:', x_test_unlab.shape, y_test_unlab.shape)
# summarize test set size
print('Test Set:', x_test.shape, y_test.shape)


print('done')



################################################################################

#DECISION TREE

#Define decision Tree
clfDT =  tree.DecisionTreeClassifier(max_depth=4)

#train the classifiers (DT)                       
clfDT.fit(x_train, y_train)

#test the trained model on the test set (DT)
y_test_pred_DT=clfDT.predict(x_test)

#Confusion matrix for DT
confMatrixTestDT=confusion_matrix(y_test, y_test_pred_DT, labels=None)

#Print out DT confusion Matrix
print ('\n Conf matrix Decision Tree')
print (confMatrixTestDT)
print ('\n')

#Measures of performance: Precision, Recall, F1
print ('Tree: Macro Precision, recall, f1-score')
print ( precision_recall_fscore_support(y_test, y_test_pred_DT, average='macro'))


# calculate score for test set
score = accuracy_score(y_test, y_test_pred_DT)
# summarize score
print('Accuracy: %.3f' % (score*100))

#use http://www.webgraphviz.com/ to visualize
tree.export_graphviz(clfDT, out_file='Classifier.dot',class_names=['<=50K','>50K'], feature_names=featurelist)

print('Tree rules=\n',tree.export_text(clfDT,feature_names=featurelist))


pr_y_test_pred_DT=clfDT.predict_proba(x_test)

#ROC curve
fprDT, tprDT, thresholdsDT = roc_curve(y_test, pr_y_test_pred_DT[:,1])

#Plot a graph for the DT
lw=2
plt.figure(10)
plt.plot(fprDT,tprDT,color='blue')
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic - DT')
plt.legend(loc="lower right")
plt.show()



################################################################################

#NAIVE BAYES

#Define a Naive Bayes
clfNB = GaussianNB()

#train the classifiers                       
clfNB.fit(x_train, y_train)

#test the trained model on the test set
y_test_pred_NB=clfNB.predict(x_test)

#Confusion matrix for NB
confMatrixTestNB=confusion_matrix(y_test, y_test_pred_NB, labels=None)

#Print out NB confusion Matrix
print ('Conf matrix Naive Bayes')
print (confMatrixTestNB)
print ()

pr_y_test_pred_NB=clfNB.predict_proba(x_test)

#ROC curve
fprNB, tprNB, thresholdsNB = roc_curve(y_test, pr_y_test_pred_NB[:,1])

# Measures of performance: Precision, Recall, F1
print ('Naive: Macro Precision, recall, f1-score')
print ( precision_recall_fscore_support(y_test, y_test_pred_NB, average='macro'))

#Plot a graph for the NB
lw=2
plt.figure(10)
plt.plot(fprNB,tprNB,color='red')
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic - NB')
plt.legend(loc="lower right")
plt.show()

################################################################################

#NEAREST NEIGHBOURS

#Define Nearest Neighbours classifiers 
#n_neighbors: is the number of neighbors
#metric: is the distance measure,default is minkowski try: mahalanobis
clfNN = KNeighborsClassifier(n_neighbors = 3)

#train the classifiers                       
clfNN.fit(x_train, y_train)

#test the trained model on the test set
y_test_pred_NN=clfNN.predict(x_test)

#Confusion matrix for NN
confMatrixTestNN=confusion_matrix(y_test, y_test_pred_NN, labels=None)

#Print out NN confusion Matrix
print ('Conf matrix Nearest Neighbor')
print (confMatrixTestNN)
print ()

#Measures of performance: Precision, Recall, F1
print ('NearNeigh: Macro Precision, recall, f1-score')
print ( precision_recall_fscore_support(y_test, y_test_pred_NN, average='macro'))
print ('\n')

pr_y_test_pred_NN=clfNN.predict_proba(x_test)

#ROC curve
fprNN, tprNN, thresholdsNN = roc_curve(y_test, pr_y_test_pred_NN[:,1])

#Plot a graph for the NN
lw=2
plt.figure(10)
plt.plot(fprNN,tprNN,color='black')
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic - NN')
plt.legend(loc="lower right")
plt.show()

################################################################################

#SUPPORT VECTOR MACHINE 

#Define Support vector machine- This has to be skipped for performance reasons. Takes ~25 minutes to perform the 'poly' kernel. same version but with 'rbf' provided bellow
#degree: refers to the degree of the polynomial kernel
# clfSVM= svm.SVC(C=100.0, cache_size=5000, class_weight=None, coef0=0.0,
#     decision_function_shape='ovo', degree=2, gamma='auto', kernel='poly',
#     max_iter=-1, random_state=None, shrinking=True,
#     tol=0.001, verbose=False, probability=True)

clfSVM= svm.SVC(C=1000.0, cache_size=2000, class_weight=None, coef0=0.0,
    decision_function_shape='ovo', degree=1, gamma='auto', kernel='rbf',
    max_iter=-1, random_state=None, shrinking=True,
    tol=0.001, verbose=False, probability=True)

#train the classifiers                       
clfSVM.fit(x_train, y_train)

#test the trained model on the test set
y_test_pred_SVM = clfSVM.predict(x_test)

#Confusion matrix for SVM
confMatrixTestSVM = confusion_matrix(y_test, y_test_pred_SVM, labels=None)

#Print out SVM confusion Matrix
print ('Conf matrix Support Vector Classifier')
print (confMatrixTestSVM)
print ()

# Measures of performance: Precision, Recall, F1
print ('Support Vector: Macro Precision, recall, f1-score')
print ( precision_recall_fscore_support(y_test, y_test_pred_SVM, average='macro'))
print ('\n')

pr_y_test_pred_SVM=clfSVM.predict_proba(x_test)


#ROC curve
fprSVM, tprSVM, thresholdsSVM = roc_curve(y_test, pr_y_test_pred_SVM[:,1])

#Plot a graph for the SVM
lw=2
plt.figure(10)
plt.plot(fprSVM,tprSVM,color='green')
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic - SVM')
plt.legend(loc="lower right")
plt.show()

################################################################################

#PLOT ALL THE CLASSIFIERS IN ONE GRAPH

lw=2
plt.figure(10)
plt.plot(fprDT,tprDT,color='blue',label='DT')
plt.plot(fprSVM,tprSVM,color='green',label='SVM')
plt.plot(fprNB,tprNB,color='red',label='NB')
plt.plot(fprNN,tprNN,color='black',label='NN')
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic - All Classifiers')
plt.legend(loc="lower right")
plt.show()

# ################################################################################

# #Plot Change in F1 Score for different number of k Nearest Neighbours

# NN_val = [] 

# for k in range(10):
#     k = k+1
    
#     clfNN = KNeighborsClassifier(n_neighbors=k)
#     clfNN.fit(x_train, y_train)
#     y_test_pred_NN=clfNN.predict(x_test)
#     confMatrixTestNN=confusion_matrix(y_test, y_test_pred_NN, labels=None)
#     NN_val.append(f1_score(y_test, y_test_pred_NN, average='macro'))
#     print('F1 score for k=',k,f1_score(y_test,y_test_pred_NN,average='macro'))




################################################################################

# Unlabeled data classification
# Semi supervised learning
# Label Propagation
# https://machinelearningmastery.com/semi-supervised-learning-with-label-propagation/



model = LabelPropagation()
X_train_mixed = np.concatenate((x_train, x_test_unlab))
y_train_mixed = np.concatenate((y_train, y_test_unlab))

model.fit(X_train_mixed, y_train_mixed)

# make predictions on hold out test set
yhat = model.predict(x_test)
# calculate score for test set
score = accuracy_score(y_test, yhat)
# summarize score
print('Accuracy: %.3f' % (score*100))

