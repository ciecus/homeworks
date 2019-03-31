import pandas as pd
import os
from numpy import *
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sbn
sbn.set(color_codes = True)
plt.rcParams['axes.unicode_minus'] = False
from scipy.stats import kstest
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
import pyecharts
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d


#load data
data1 = pd.read_csv('clean1.data',header = None)
target1 = data1[168].values
y1 = pd.get_dummies(data1[168]).values
data1.drop([1,0,168],axis =1,inplace = True)
data1 = data1.values


#self pca
def pca(x,d):
    meanValue=np.mean(x,0)
    x = x-meanValue
    cov_mat = np.cov(x,rowvar = 0)
    eigVal, eigVec = np.linalg.eig(mat(cov_mat))
    #取最大的d个特征值
    eigValSorted_indices = np.argsort(eigVal)
    eigVec_d = eigVec[:,eigValSorted_indices[:-d-1:-1]] #-d-1前加:才能向左切
    eigVal_d = eigVal[eigValSorted_indices[:-d-1:-1]]
    contributions = round(float(sum(eigVal_d)/sum(eigVal)),2)
    #print("----------------------------eig vectors----------------\n",eigVec_d)
    #print("----------------------------eig values----------------\n",eigVal_d)
    #print("----------------------------contributions----------------\n",contributions)
    return eigVec_d,eigVal_d,contributions

def pca_contribution_plt(data,max_dim):
    contributions_list = []
    for n in range(max_dim):
        #print("the eigenvalue number is %d\n"%n)
        eigVec_d,eigVal_d,contributions = pca(data,n)
        contributions_list.append(contributions)
    contribution_growth = [round(contributions_list[i]-contributions_list[i-1],2) for i in range(1,max_dim)]
    plt.figure()
    plt.plot(range(max_dim),contributions_list)
    plt.plot(range(1,max_dim),contribution_growth)
    plt.show()
    return contributions_list,contribution_growth

def plot(data,max_dim):
    contributions_list,contribution_growth = pca_contribution_plt(data,max_dim)
    data_scale = preprocessing.scale(data)
    contributions_list_scale,contribution_growth_scale = pca_contribution_plt(data_scale,max_dim)
    return contributions_list,contribution_growth,contributions_list_scale,contribution_growth_scale

def pca_train(data,max_dim):
    meanValue=np.mean(data,0)
    data = data-meanValue
    for n in range(max_dim):
            eigVec_d,eigVal_d,contributions = pca(data,n)
            if contributions>0.85:
                break
    newdata=np.dot(data,eigVec_d)
    return eigVec_d,eigVal_d,contributions,newdata

#self pca result
print('------------------------------self pca ----------------------------')
eigVec_d,eigVal_d,contributions,newdata = pca_train(data1,max_dim=50)
print("the number of eigVec",len(eigVal_d))
print("eigVec\n",eigVec_d,"\neigVal\n",eigVal_d,"\ncontributions\n",contributions)
print(newdata)

#sklear pca
from sklearn.decomposition import PCA
meanValue=np.mean(data1,0)
data1 = data1-meanValue
pca_sklearn=PCA(n_components=3).fit(data1)
reducedx = pca_sklearn.fit_transform(data1)
print('-------------------------------------skleran pca ---------------------------------')
print("explained_variance_ratio:",pca_sklearn.explained_variance_ratio_)
print("singular_values",pca_sklearn.singular_values_)
print(reducedx)


#svd pca
meanValue=np.mean(data1,0)
data1 = data1-meanValue
u, sigma, v = np.linalg.svd(data1[:, :])
#svd_pca_new_data = np.dot(u[:,:13],np.diag(sigma)[:13,:13])
svd_pca_new_data = np.dot(data1,u[:,:13])
print('-------------------------------------svd pca ---------------------------------')
print(sigma[:13])
print(svd_pca_new_data)


#visualize 2d
def visualize_plot_2D(reduced_x,y):
    red_x,red_y=[],[]
    blue_x,blue_y=[],[]
    for i in range(len(reduced_x)):
        if y[i] ==0:
            red_x.append(reduced_x[i][0])
            red_y.append(reduced_x[i][1])

        else:
            blue_x.append(reduced_x[i][0])
            blue_y.append(reduced_x[i][1])
    plt.scatter(red_x,red_y,c='r',marker='x')
    plt.scatter(blue_x,blue_y,c='b',marker='D')
    plt.title('pca visualize 2d')
    plt.show()
visualize_plot_2D(svd_pca_new_data,target1)


#visualize 3d
def visualize_plot_3D(reduced_x,y):
    red_x,red_y,red_z=[],[],[]
    blue_x,blue_y,blue_z=[],[],[]
    for i in range(len(reduced_x)):
        if y[i] ==0:
            red_x.append(reduced_x[i][0])
            red_y.append(reduced_x[i][1])
            red_z.append(reduced_x[i][2])

        else:
            blue_x.append(reduced_x[i][0])
            blue_y.append(reduced_x[i][1])
            blue_z.append(reduced_x[i][2])
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111, projection='3d')
    plt.rcParams['legend.fontsize'] = 10
    ax.plot(red_x,red_y,red_z, 'o', markersize=8, color='blue', alpha=0.5, label='class1')
    ax.plot(blue_x,blue_y,blue_z, '^', markersize=8, alpha=0.5, color='red', label='class2')

    plt.title('pca visualize 3d')
    ax.legend(loc='upper right')

    plt.show()
visualize_plot_3D(svd_pca_new_data,target1)


#use low dimension for classification
import tensorflow as tf
from sklearn.model_selection import KFold
import pandas as pd
def softmax_function(x_train,y_train,x_test,y_test):
    loss_list = []
    x = tf.placeholder(tf.float32,shape = (None,13))
    y = tf.placeholder(tf.float32,shape = (None,2)) #predict
    #用softmax 构建模型
    w = tf.Variable(tf.zeros([13,2]))
    b = tf.Variable(tf.zeros([2]))
    pred = tf.nn.softmax(tf.matmul(x,w)+b)
    #损失函数（交叉熵）
    with tf.name_scope('loss'):
        loss = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred),1))
        tf.summary.scalar('loss',loss)
    #梯度下降
    optimizer = tf.train.GradientDescentOptimizer(0.05).minimize(loss)
    #准确率
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1)), tf.float32))


    #加载session 图
    with tf.Session() as sess:
        #初始化所有变量
        init = tf.global_variables_initializer()
        sess.run(init)
        #开始训练
        for epoch in range(10000):
            sess.run(optimizer,feed_dict={x:x_train,y:y_train})
            loss_list.append(sess.run(loss,feed_dict={x:x_train,y:y_train}))
        print('train accuracy:',sess.run(accuracy,feed_dict={x:x_train,y:y_train}))
        print('test accuracy:',sess.run(accuracy,feed_dict={x:x_test,y:y_test}))
        plt.figure()
        plt.plot(range(10000),loss_list)

X = svd_pca_new_data
Y = y1
KF=KFold(n_splits=5)
i = 1
for train_index,test_index in KF.split(X):

    print('for %d fold'%i)
    i+=1
    X_train,X_test=X[train_index],X[test_index]
    Y_train,Y_test=Y[train_index],Y[test_index]
    softmax_function(X_train,Y_train,X_test,Y_test)