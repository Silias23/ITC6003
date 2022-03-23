import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mping
#import seaborn as sns
from scipy.cluster.hierarchy import dendrogram
from scipy.cluster import hierarchy
#import statsmodels.api as sm
#import statsmodels.formula.api as smf
from sklearn.model_selection import train_test_split,cross_val_score,cross_val_predict,ShuffleSplit,GridSearchCV
from sklearn.cluster import KMeans,AgglomerativeClustering,DBSCAN
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.datasets import make_blobs,make_moons
from sklearn.metrics import silhouette_samples
#from yellowbrick.cluster import KElbowVisualizer
import time
from matplotlib.colors import ListedColormap
#from skompiler import skompile
from joblib import dump, load
from sklearn.preprocessing import MinMaxScaler
from sklearn import tree
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score




data = pd.read_csv("ITC6003Project/data.csv", delimiter=",")


data['Channel'].value_counts()
data['Region'].value_counts()

# data['Channel'] = data['Channel'].replace(1,"H")
# data['Channel'] = data['Channel'].replace(2,"R")
# data['Region'] = data['Region'].replace(1,"L")
# data['Region'] = data['Region'].replace(2,"O")
# data['Region'] = data['Region'].replace(3,"Other")

# data = pd.get_dummies(data,columns=['Channel','Region'],prefix=['Channel','Region'])


data = data.drop('Channel', axis= 1)
data = data.drop('Region', axis= 1)

minmax_scale = MinMaxScaler().fit_transform(data)

scaled_frame2 = pd.DataFrame(minmax_scale,columns=data.columns)

scaled_frame2.head()


X=np.array(scaled_frame2)


## Clustering
#run clusterings for differen values of k
inertiasAll=[]
silhouettesAll=[]
for n in range(2,12):
    print ('Clustering for n=',n)
    kmeans = KMeans(n_clusters=n)
    kmeans.fit(X)
    y_kmeans = kmeans.predict(X)

#get cluster centers
    kmeans.cluster_centers_

#evalute
    print ('inertia=',kmeans.inertia_)
    silhouette_values = silhouette_samples(X, y_kmeans)
    print ('silhouette=', np.mean(silhouette_values))
    
    inertiasAll.append(kmeans.inertia_)
    silhouettesAll.append(np.mean(silhouette_values))    



#Print some statistical data: intertia and silhouetter
plt.figure(1)
plt.plot(range(2,12),silhouettesAll,'r*-')
plt.ylabel('Silhouette score')
plt.xlabel('Number of clusters')
#plt.show()
plt.figure(2)
plt.plot(range(2,12),inertiasAll,'g*-')
plt.ylabel('Inertia Score')
plt.xlabel('Number of clusters')
#plt.show()


opt_clusters = 6

kmeans = KMeans(n_clusters=opt_clusters)
kmeans.fit(X)
print(kmeans.cluster_centers_)

print('\nFeatures names ',data.columns)


for i in range(opt_clusters):
    print(f'\nCluster Centre {i+1}',kmeans.cluster_centers_[i])

for i in range(opt_clusters):
    print(f'\nSize of Cluster-{i+1}=', len(kmeans.labels_[kmeans.labels_==i]))

idxCluster = []
for i in range(opt_clusters):
    idxCluster.append([])
    idxCluster[i] = np.where(kmeans.labels_==i)

for i in range(opt_clusters):
    plt.figure(i)
    plt.title(f'Cluster-{i}')
    # x-labels are turned 90 degrees
    plt.xticks(rotation=90)
    plt.boxplot(X[idxCluster[i]],labels=list(data.columns))
    plt.show()


#insert the clusters as class labels to prepare for decision tree analysis
#copy df to clean copy
x = scaled_frame2.copy()
# y are the clusters from clustering
y = pd.DataFrame(kmeans.labels_)
#preprocess to scale further and normalize data - might not be needed, have to ask
x1 = preprocessing.scale(x)


#!!!! play with max_depth
clf = tree.DecisionTreeClassifier(criterion="gini", random_state=0, max_depth= 4)

# fit and train the model with all the data - no need to split and test/train for accuracy - we already know the results and just want to get the logic
clf = clf.fit(x1, y)

#run prediction
Y_train_pred=clf.predict(x1)

#evaluate prediction with true positives
confMatrixTrain = confusion_matrix(y, Y_train_pred, labels=None)
print(confMatrixTrain)

#use http://www.webgraphviz.com/ to visualize
tree.export_graphviz(clf, out_file='iris-tree.dot',class_names=['Cluster1','Cluster2','Cluster3','Cluster4','Cluster5','Cluster6'], feature_names=list(data.columns.values))

print('Tree rules=\n',tree.export_text(clf,feature_names=list(data.columns.values)))

print ('\tClassifier Evaluation')
print ('Accuracy Train=', accuracy_score(y, Y_train_pred, normalize=True))
# Macro: 
print('Macro-train f1=',f1_score(y, Y_train_pred, average='macro')) 
print('Micro-train f1=',f1_score(y, Y_train_pred, average='micro')) 
print('train f1 per class=',f1_score(y, Y_train_pred, average=None))
print ('Macro: train-Precision-Recall-FScore-Support',precision_recall_fscore_support(y, Y_train_pred, average='macro'))

# Some statistics on the Tree:
print('Max Depth=',clf.get_depth())
print('Number of Leaves=',clf.get_n_leaves())




#areas to improve:
# instead of using MinMax and preprocessing scale, find an other way to normalize the dataset in order to be able to produce a decision tree that
#shows actual customer data thresholds and not downscaled numbers

#hierarchical clustering
