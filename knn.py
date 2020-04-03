# -*- python3.8
from os import listdir
from numpy import *
import numpy as np
import operator
import datetime


def KNN(test_data, train_data, train_label, k):
    # 已知分类的数据集（训练集）的行数
    dataSetSize = train_data.shape[0]
    # 求所有距离：先tile函数将输入点拓展成与训练集相同维数的矩阵，计算测试样本与每一个训练样本的距离
    all_distances = np.sqrt(np.sum(np.square(tile(test_data, (dataSetSize, 1)) - train_data), axis=1))
    # print("所有距离：",all_distances)
    # 按all_distances中元素进行升序排序后得到其对应索引的列表
    sort_distance_index = all_distances.argsort()
    # print("文件索引排序：",sort_distance_index)
    # 选择距离最小的k个点
    classCount = {}
    for i in range(k):
        # 返回最小距离的训练集的索引(预测值)
        voteIlabel = train_label[sort_distance_index[i]]
        # print('第',i+1,'次预测值',voteIlabel)
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    # 求众数：按classCount字典的第2个元素（即类别出现的次数）从大到小排序
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


# 文本向量化 32x32 -> 1x1024
def img2vector(filename):
    returnVect = []
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect.append(int(lineStr[j]))
    return returnVect


# 从文件名中解析分类数字
def classnumCut(fileName):
    # 参考文件名格式为：0_3.txt
    fileStr = fileName.split('.')[0]
    classNumStr = int(fileStr.split('_')[0])
    return classNumStr


# 构建训练集数据向量，及对应分类标签向量
def trainingDataSet():
    train_label = []
    trainingFileList = listdir('trainingDigits')
    m = len(trainingFileList)
    train_data = zeros((m, 1024))
    # 获取训练集的标签
    for i in range(m):
        # fileNameStr:所有训练集文件名
        fileNameStr = trainingFileList[i]
        # 得到训练集索引
        train_label.append(classnumCut(fileNameStr))
        train_data[i, :] = img2vector('trainingDigits/%s' % fileNameStr)
    return train_label, train_data


# 测试函数
def main():
    t1 = datetime.datetime.now()  # 计时开始
    Nearest_Neighbor_number = int(input('选取最邻近的K个值，K='))
    train_label, train_data = trainingDataSet()
    testFileList = listdir('testDigits')
    error_sum = 0
    test_number = len(testFileList)
    for i in range(test_number):
        # 测试集文件名
        fileNameStr = testFileList[i]
        # 切片后得到测试集索引
        classNumStr = classnumCut(fileNameStr)
        test_data = img2vector('testDigits/%s' % fileNameStr)
        # 调用knn算法进行测试
        classifierResult = KNN(test_data, train_data, train_label, Nearest_Neighbor_number)
        print("第", i + 1, "组：", "预测值:", classifierResult, "真实值:", classNumStr)
        if (classifierResult != classNumStr):
            error_sum += 1.0
    print("\n测试集总数为:", test_number)
    print("测试出错总数:", error_sum)
    print("\n错误率:", error_sum / float(test_number) * 100, '%')
    t2 = datetime.datetime.now()
    print('耗 时 = ', t2 - t1)


if __name__ == "__main__":
    main()
