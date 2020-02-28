import numpy as np
import math
from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold
import pandas
from scipy.io import arff
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn import metrics
from sklearn.model_selection import GridSearchCV, train_test_split

def getdata(data):
    instance = data.shape[0]
    attribute = data.shape[1] - 1
    Defective = data[:, attribute]
    data = np.delete(data, attribute, axis=1)
    return data,instance,attribute,Defective

def getfeature(feature,fRank,n):
    Data = []
    for i in range(n):
         Data.append(feature[fRank[i]])
    return Data

def getDefectiveNum(Defective):
    DefectiveNum = []
    for i,v in enumerate(Defective):
        if v == b'Y':
            DefectiveNum.append(0)
        else:
            DefectiveNum.append(1)
    DefectiveNum = [float(i) for i in DefectiveNum]
    return DefectiveNum

def getminposition(mylist):
    minnum = min(mylist)
    position = mylist.index(minnum)
    return position

def delete(list1,list2):
    deltalist = []
    for i,v in enumerate(list1):
        deltalist.append(abs(list1[i]-list2[i]))
    return deltalist

def add(list1,list2):
    sumlist = []
    for i,v in enumerate(list1):
        sumlist.append(abs(list1[i]+list2[i]))
    return sumlist

def DecisionTree(X_train,Y_train):
    tree = DecisionTreeClassifier(max_depth=5, random_state=0)
    tree.fit(X_train, Y_train)
    return tree

def SVM(X_train, Y_train):
    lsvc = LinearSVC(max_iter=2000)
    lsvc.fit(X_train, Y_train)
    return lsvc

def F_Measure(true,pred):
    P = (metrics.precision_score(true, pred))
    R = (metrics.recall_score(true, pred))
    F = 2 * P * R / (P + R)
    return F

def F(TP, FN, FP, TN):
    if TP == 0: TP = 1
    P = TP / (TP + FP)
    R = TP / (TP + FN)
    F = 2 * P * R / (P + R)
    return F

filename = './Data/CM1'

dataset, mate = arff.loadarff(filename+'.arff')
#print(dataset)
df = pandas.DataFrame(dataset)
#print(df)
#dataset_ls = list(dataset)
originData = np.array(df)
#print(originData.shape)
data, instance, attribute, Defective = getdata(originData)
# print(data, "\n",instance,"\n", attribute,"\n", Defective)
#print(Tdata)
DefectiveNum = getDefectiveNum(Defective)
# print(DefectiveNum)
#标准化
data = data.astype('float64')
scaled = preprocessing.scale(data)
# print(scaled)

#计算k
k = np.sum(Defective == b'N')/np.sum(Defective == b'Y')
k = int(k)

# print(k)
#标准化距离
ed = np.zeros((instance, instance))
alldata = instance*instance
for i in range(0, instance):
    for j in range(0, instance):
        ed[i][j] = np.linalg.norm(scaled[i,:] - scaled[j,:])
# print(ed)

#临近样本
rank = []
for i in range(0, instance):
    rank.append(pandas.Series(ed[i, :]).rank(method='min').tolist())
#print(rank)

nearest = []

for index,i in enumerate(rank):
    n = []
    num = 0
    while 1:
        position = getminposition(i)
        if Defective[position] == b'Y' or position == index:
            i[position] = max(i)
        else:
            n.append(position)
            i[position] = max(i)
            num += 1
        if num == k:
            break
    nearest.append(n)
# print(nearest)

#特征差值
delta = []
for i,v in enumerate(data):
        d = []
        for j,w in enumerate(nearest[i]):
            d.append(delete(data[i], data[w]))
        delta.append(d)
#print(delta)

#特征权重
W = np.zeros(attribute)
#print(W)
for i,v in enumerate(delta):
    if Defective[i] == b'Y':
        for j in v:
            W = add(W,pandas.Series(j).rank(method='min').tolist())
# print(W)

#特征排序列表
WRank = pandas.Series(W).rank(method='min').tolist()
fRank = []
flag = 0
while 1:
    for i, v in enumerate(WRank):
        if v == max(WRank):
            fRank.append(i)
            WRank[i] = -1
            flag += 1
        if flag == attribute:
            break
    if flag == attribute:
        break

print(filename+"特征排序列表:",fRank)

skf = StratifiedKFold(n_splits=10, random_state=None, shuffle=True)
# skf.get_n_splits(data, DefectiveNum)
# print(skf)
DefectiveNum = np.array(DefectiveNum)
for train_index, test_index in skf.split(scaled, DefectiveNum):
    # print(type(train_index[1]))
    # print("TRAIN:", train_index,'\n', "TEST:", test_index)
    All_X_train, All_X_test = scaled[train_index], scaled[test_index]
    Y_train, Y_test = np.array(DefectiveNum)[train_index], DefectiveNum[test_index]
    log2_X_train,log2_X_test = [], []
    for i in range(len(All_X_train)):
        log2_X_train.append(getfeature(All_X_train[i],fRank,int(math.log2(All_X_train.shape[1]))))
    for i in range(len(All_X_test)):
        log2_X_test.append(getfeature(All_X_test[i],fRank,int(math.log2(All_X_test.shape[1]))))
    # print(len(log2_X_train),len(log2_X_train[0]),len(log2_X_test),len(log2_X_test[0]))
    log2_tree = DecisionTree(log2_X_train,Y_train)
    All_tree = DecisionTree(All_X_train,Y_train)

    log2_SVM = SVM(log2_X_train,Y_train)
    All_SVM = SVM(All_X_train,Y_train)

    log2_tree_pred = log2_tree.predict(log2_X_test)
    All_tree_pree = All_tree.predict(All_X_test)

    log2_SVM_pred = log2_SVM.predict(log2_X_test)
    All_SVM_pred = All_SVM.predict(All_X_test)
    #TP_log2_tree, FN_log2_tree, FP_log2_tree, TN_log2_tree
    C = metrics.confusion_matrix(Y_test, log2_tree_pred).ravel()
    TP_log2_tree, FN_log2_tree, FP_log2_tree, TN_log2_tree = metrics.confusion_matrix(Y_test, log2_tree_pred).ravel()
    TP_All_tree, FN_All_tree, FP_All_tree, TN_All_tree = metrics.confusion_matrix(Y_test, All_tree_pree).ravel()
    TP_log2_SVM, FN_log2_SVM, FP_log2_SVM, TN_log2_SVM = metrics.confusion_matrix(Y_test, log2_SVM_pred).ravel()
    TP_All_SVM, FN_All_SVM, FP_All_SVM, TN_All_SVM = metrics.confusion_matrix(Y_test, All_tree_pree).ravel()
    #print(TP_log2_tree)
    #print(FP_log2_tree)
    #print(FN_log2_tree)
    F_log2_tree = F(TP_log2_tree, FN_log2_tree, FP_log2_tree, TN_log2_tree)
    F_All_tree = F(TP_All_tree, FN_All_tree, FP_All_tree, TN_All_tree)
    F_log2_SVM = F(TP_log2_SVM, FN_log2_SVM, FP_log2_SVM, TN_log2_SVM)
    F_All_SVM = F(TP_All_SVM, FN_All_SVM, FP_All_SVM, TN_All_SVM)

    #F_log2_tree = F_Measure(Y_test,log2_tree_pred)
    #F_All_tree = F_Measure(Y_test,All_tree_pree)

    #F_log2_SVM = F_Measure(Y_test,log2_SVM_pred)
    #F_All_SVM = F_Measure(Y_test,All_tree_pree)

    print(filename+"决策树log2子集F1值:",F_log2_tree,"决策树原始特征集F1值:",F_All_tree,'\n',
          filename+"SVM训练log2子集F1值:",F_log2_SVM,"SVM训练原始特征集F1值:",F_All_SVM)



