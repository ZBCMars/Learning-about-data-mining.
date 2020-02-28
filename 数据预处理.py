import numpy as np
from sklearn import preprocessing
from sklearn import tree
from sklearn import svm
from sklearn import naive_bayes
from sklearn.model_selection import RepeatedKFold
from sklearn import metrics
import pandas
import copy, os, sys, arff
from openpyxl import Workbook
from scipy.io import arff


def restart_program():
  python = sys.executable
  os.execl(python, python, * sys.argv)


def getdata(data):
    instance = data.shape[0]
    attribute = data.shape[1] - 1
    Defective = data[:, attribute]
    data = np.delete(data, attribute, axis=1)
    return data,instance,attribute,Defective


def getDefectiveNum(Defective):
    DefectiveNum = copy.deepcopy(Defective)
    for i,v in enumerate(DefectiveNum):
        if v == 'Y':
            DefectiveNum[i] = 0
        else:
            DefectiveNum[i] = 1
    DefectiveNum = [float(i) for i in DefectiveNum]
    return DefectiveNum


def DT(train_Data,train_Defective,test_Data,test_Defective):
    clf = tree.DecisionTreeClassifier()
    clf.fit(train_Data, train_Defective)
    predict_prob = clf.predict_proba(test_Data)[:, 1]
    DT_auc = metrics.roc_auc_score(getDefectiveNum(test_Defective), predict_prob)
    return DT_auc


def SVC(train_Data,train_Defective,test_Data,test_Defective):
    clf = svm.SVC(C=4)#(gamma='rbf'
    train_Defective = getDefectiveNum(train_Defective)
    clf.fit(train_Data, train_Defective)
    pred = clf.predict(test_Data)
    #pred = getDefectiveNum(pred)
    test_Defective = getDefectiveNum(test_Defective)
    SVC_auc = metrics.roc_auc_score(test_Defective, pred)
    return SVC_auc
    """svc = svm.SVC()
    svc.fit(train_Data, train_Defective)
    predict_prob = svc.predict(test_Data)
    TP = 0  # 正确预测为正例数
    FP = 0  # 错误预测为正例数
    FN = 0  # 错误预测为负例数

    for index, predict in enumerate(predict_prob):
        if predict == 'Y' and test_Defective[index] == 'Y':
            TP += 1
        if predict == 'Y' and test_Defective[index] == 'N':
            FP += 1
        if predict == 'N' and test_Defective[index] == 'Y':
            FN += 1

    if TP + FP == 0 or TP + FN == 0:
        SVC_F1 = 0
    else:
        P = TP / (TP + FP)
        R = TP / (TP + FN)
        if P + R == 0:
            SVC_F1 = 0
        else:
            SVC_F1 = 2 * P * R / (P + R)
    return SVC_F1"""

def func(myset):
    test_auc = 0
    test_F1 = 0
    try:
        kf = RepeatedKFold(n_splits=10, n_repeats=10)
        for train_index, test_index in kf.split(myset):
            train = []
            test = []
            for train_num in train_index:
                train.append(originData[train_num])
            for test_num in test_index:
                test.append(originData[test_num])
        train = np.array(train)
        test = np.array(test)
        train_Data, train_Instance, train_Attribute, train_Defective = getdata(train)
        test_Data, test_Instance, test_Attribure, test_Defective = getdata(test)
        test_auc = DT(train_Data, train_Defective, test_Data, test_Defective)

        test_F1 = SVC(train_Data, train_Defective, test_Data, test_Defective)
    except:
        func(myset)
    return test_auc,test_F1


'''新建数据表'''
wb = Workbook()
ws = wb.active

'''fileList = ['CM1', 'KC1', 'KC3', 'MC2', 'MW1', 'PC1', 'PC2', 'PC3', 'PC4', 'JM1', 'MC1']#, 'PC5']'''
'''JM1 MC1 PC5'''

'''for filename in fileList:'''
filename = 'CM1'
ws.append([filename])

'''导入数据'''
dataset, mate = arff.loadarff(filename+'.arff')
#dataset = arff.load(open(filename+'.arff'))
#dataset_list = list(dataset)
#originData = np.array(dataset)
dataset = pandas.DataFrame(dataset)
originData = np.array(dataset)
data, instance, attribute, Defective = getdata(originData)
DefectiveNum = getDefectiveNum(Defective)

'''标准化'''
data = data.astype('float64')
scaled = preprocessing.scale(data)
#print(scaled)

'''计算k'''
'''k = np.sum(Defective == "N")/np.sum(Defective == "Y")
if k == k:
    k = int(k)
else:
    k = 0'''

k = np.sum(Defective == b'N')/np.sum(Defective == b'Y')
k = int(k)

#print(k)
'''标准化距离'''
ed = np.zeros((instance, instance))
alldata = instance*instance
for i in range(0, instance):
    for j in range(0, instance):
        caleddata = i*instance + j
        percent = int(caleddata/alldata*100)
        #ed[i][j] = np.sqrt(np.sum(np.square(scaled[i, :] - scaled[j, :])))
        print("\r计算"+filename+"特征排序列表:"+"."*percent+str(percent)+"%",end=' ')
        ed[i][j] = np.linalg.norm(scaled[i,:] - scaled[j,:])
print()
#print(ed)

'''临近样本'''
rank = []
for i in range(0, instance):
    rank.append(pandas.Series(ed[i, :]).rank(method='min').tolist())
#print(rank)

nearest = []
rankcpy = copy.deepcopy(rank)


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


for index,i in enumerate(rankcpy):
    n = []
    num = 0
    while 1:
        position = getminposition(i)
        if Defective[position] == 'Y' or position == index:
            i[position] = max(i)
        else:
            n.append(position)
            i[position] = max(i)
            num += 1
        if num == k:
            break
    nearest.append(n)
#print(nearest)

'''特征差值'''
delta = []
for i,v in enumerate(data):
        d = []
        for j,w in enumerate(nearest[i]):
            d.append(delete(data[i], data[w]))
        delta.append(d)
#print(delta)

'''特征权重'''
W = np.zeros(attribute)
#print(W)
for i,v in enumerate(delta):
    if Defective[i] == 'Y':
        for j in v:
            W = add(W,pandas.Series(j).rank(method='min').tolist())
#print(W)

'''特征排序列表'''
WRank = pandas.Series(W).rank(method='min').tolist()
fRank = []
WRankcpy = copy.deepcopy(WRank)
#print(WRankcpy)
flag = 0
while 1:
    for i, v in enumerate(WRankcpy):
        if v == max(WRankcpy):
            fRank.append(i+1)
            WRankcpy[i] = -1
            flag += 1
        if flag == attribute:
            break
    if flag == attribute:
        break

print(filename+"特征排序列表")
print(fRank)


def getSet(data,rank,num):
    subset = []
    for i in range(instance):
        subset.append([])
    for j in range(num):
        for i,v in enumerate(data[:,rank[j]-1]):
            subset[i].append(v)
    for i,v in enumerate(Defective):
        subset[i].append(v)
    return subset

AUCList = []
F1List = []
for i in range(1,attribute+1):
    myset = getSet(data,fRank,i)
    auc,F1 = func(myset)
    AUCList.append(auc)
    F1List.append(F1)

ws.append(['决策树-AUC'])
ws.append(AUCList)
ws.append(['最大值位置'])
AUCindex = AUCList.index(max(AUCList))+1
ws.append([AUCindex])
ws.append(["对应序列"])
ws.append(fRank[0:AUCindex])
ws.append(['SVM-AUC'])
ws.append(F1List)
ws.append(['最大值位置'])
Findex = F1List.index(max(F1List))+1
ws.append([Findex])
ws.append(["对应序列"])
ws.append(fRank[0:Findex])
print(filename+'finish')

#wb.save('Data/test.xlsx')