import numpy as np
import math
from sklearn import preprocessing
import pandas
from scipy.io import arff

def getdata(data):
    instance = data.shape[0]
    attribute = data.shape[1] - 1
    Defective = data[:, attribute]
    data = np.delete(data, attribute, axis=1)
    return data,instance,attribute,Defective

def getDefectiveNum(Defective):
    DefectiveNum = []
    for i,v in enumerate(Defective):
        if v == b'Y':
            DefectiveNum.append(0)
        else:
            DefectiveNum.append(1)
    DefectiveNum = [float(i) for i in DefectiveNum]
    return DefectiveNum


#找到一个元素aim在列表List里全部的位置  并将所有的索引值打包成列表并返回
def findAllIndexInList(aim, List):
    pos = 0  # pos代表List里each = aim时候each的索引值
    index = []  # 所有pos打包到index里
    for each in List:
        if each == aim:
            index.append(pos)
        pos += 1
    return index

#把一个列表List里索引在Index里的元素取出来，元素组成一个 newList列表
def CreateNewListByIndex(Index, List):
    newList = []
    List = list(List)
    Index = list(Index)
    for each in Index:
        newList.append(List[each])
    return newList

#对一个列表List里的某个元素aim求概率 p(X=aim): 即 aim在List有多少个重复值/List元素总个数
def Pi(aim, List):
    length = len(list(List))  #求List一共有多少个元素，包括重复值
    aimcount = (list(List)).count(aim)	 #求aim在List里有多少个一样的值
    pi = (float)(aimcount/length)
    return pi

def entropy(data):  # 输入的data 是 X 所有取值（重复值不去除）的列表

    data1 = np.unique(data)

    resultEn = 0  # 单个元素的熵H(X)保存在resultEn

    for each in data1:  # data1里保存的值不重复
        pi = Pi(each, data)  # 求出data（data里的值可能重复）中每个 xi出现的概率
        resultEn -= pi * math.log(pi, 2)  # 对不同xi的信息熵求和过程
    return resultEn

# conditionalEntropy 求条件熵 H(X|Y) = Σp(yi)*H(X|Y=yi)
def conditionalEntropy(dataX, dataY):
    # 先对dataY处理：
    # YElementsKinds是所有原先dataY列表里不重复的元素组成的新列表：
    YElementsKinds = list(np.unique(dataY))
    resultConEn = 0  # 最终条件熵H(X|Y)
    # 在每个不同的yi = uniqueYEle 条件下:
    for uniqueYEle in YElementsKinds:
        YIndex = findAllIndexInList(uniqueYEle, dataY)
        # 找出dataY 里所有等于yi = uniqueYEl的索引值组成的列表
        dataX_Y = CreateNewListByIndex(YIndex, dataX)  # 找到dataX里所有索引在YIndex里的值组成一个列表
        HX_uniqueYEle = entropy(dataX_Y)  # 此时可以计算 HX_uniqueYEle =  H（X|Y=yi)
        pi = Pi(uniqueYEle, dataY)  # 此时可以计算 pi = p(Y=yi)
        resultConEn += pi * HX_uniqueYEle  # 求和 H（X|Y）= Σ p(Y=yi)*H（X|Y=yi)
    return resultConEn  # 返回条件熵 H（X|Y）

def transpose(M):
    # 直接使用zip解包成转置后的元组迭代器，再强转成list存入最终的list中
    return [list(row) for row in zip(*M)]

fileList = ['CM1']

for filename in fileList:
    '''导入数据'''
    dataset, mate = arff.loadarff(filename+'.arff')
    #print(dataset)
    df = pandas.DataFrame(dataset)
    originData = np.array(df)
    data, instance, attribute, Defective = getdata(originData)
    Tdata = transpose(data)
    DefectiveNum = getDefectiveNum(Defective)

    '''标准化'''
    data = data.astype('float64')
    scaled = preprocessing.scale(data)

    '''计算k'''
    k = np.sum(Defective == b'N')/np.sum(Defective == b'Y')
    k = int(k)

    '''标准化距离'''
    ed = np.zeros((instance, instance))
    alldata = instance*instance
    for i in range(0, instance):
        for j in range(0, instance):
            ed[i][j] = np.linalg.norm(scaled[i,:] - scaled[j,:])
    #print(ed)

    '''临近样本'''
    rank = []
    for i in range(0, instance):
        rank.append(pandas.Series(ed[i, :]).rank(method='min').tolist())
    #print(rank)

    nearest = []

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
        if Defective[i] == b'Y':
            for j in v:
                W = add(W,pandas.Series(j).rank(method='min').tolist())
    #print(W)

    '''特征排序列表'''
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
    M = []
    for i in range(attribute):
        X = []
        for j in range(attribute):
            HX = entropy(Tdata[i])
            HY = entropy(Tdata[j])
            HXY = conditionalEntropy(Tdata[i],Tdata[j])
            IG = HX - HXY
            SU = 2 * IG / (HX + HY)
            #print(SU)
            X.append(SU)
        M.append(X)
    # M = np.array(M)
    # print(M.shape)
    print(M)
