# encoding:utf-8
import net_func as net_util
import network as nt
import numpy as np
import math
import time
import os
from sklearn.cluster import KMeans
from collections import Counter
import itertools as it

'''
graphPathList: aspects的路径
base_addr:
cascade_name: multi-aspects的路径
numberNodes: =beta?
numberAspects: aspects的个数 (cat_num)
cascade_num: multi-aspects 的结点数 = numberAspects * beta
beta: 每个aspect的结点数
'''
def load_data_K(graphPathList, base_addr, cascade_name, numberNodes, numberAspects, cascade_num, beta):
    groundTGraphs = []
    for k in range(numberAspects):
        with open(graphPathList[k],'r') as f:
            lines = f.readlines()
            lines = [line.split() for line in lines]
            data = np.array([[int(node) for node in line] for line in lines])
            groundTruthGraph = np.zeros((numberNodes,numberNodes))
            numberEdges = data.shape[0]
            for i in range(numberEdges):
                groundTruthGraph[data[i,0]-1,data[i,1]-1]=1   # 有边则为1
            groundTGraphs.append(groundTruthGraph)  # numberAspects个图的结构，即边


    cascades = nt.m_cascade(numberNodes, cascade_num, base_addr=base_addr, cascade_name=cascade_name,
                             cat_num=numberAspects)
    cascades.read_cascade()    # 组合图的感染记录
    ground_truth_labels = np.ones([cascade_num])
    # for i in range(int(cascade_num)):
    #     ground_truth_labels[i] = int(i /beta)    # 第i条记录原本属于哪一个aspect --> 真实的aspect label
    ground_truth_labels[:beta[0]] = 0
    ground_truth_labels[beta[0]:beta[0]+beta[1]] = 1
    ground_truth_labels[beta[0]+beta[1]:beta[0]+beta[1]+beta[2]] = 2
    cascades.init_ground_truth_labels(ground_truth_labels)
    return groundTGraphs, cascades   # 返回 真实图，组合图感染记录

def cal_F1_average_mulGraph(networks, groundTGraphs, numberAspects):
    # k个子图，分别计算F1，求平均，不合并
    recall_avg=0
    precision_avg=0
    f1_avg = 0
    f1Mat = np.zeros((numberAspects,numberAspects))
    recallMat = np.zeros((numberAspects, numberAspects))
    precisionMat = np.zeros((numberAspects, numberAspects))
    record_f1 = []
    record_recall = []
    record_precision = []


    tmpList = [i for i in range(numberAspects)]
    permus = list(it.permutations(tmpList))
    epsilon = 1e-16
    for k in range(numberAspects):
        inferGraph = networks[k].graph
        for i in range(numberAspects):
            groundTruthGraph = groundTGraphs[i]
            TP = np.sum((inferGraph + groundTruthGraph) == 2)
            FP = np.sum((inferGraph - groundTruthGraph) == 1)
            FN = np.sum((inferGraph - groundTruthGraph) == -1)

            recallMat[k,i] = TP / (TP + FN + epsilon)
            precisionMat[k,i] = TP / (TP + FP + epsilon)
            f1Mat[k,i] = 2 * recallMat[k,i] * precisionMat[k,i] / (recallMat[k,i] + precisionMat[k,i]+ epsilon)

    maxSum = -np.inf
    maxIndex = -1
    for i in range(len(permus)):
        curPer = permus[i]
        sum = 0
        for k in range(numberAspects):
            sum+=f1Mat[k,curPer[k]]
        if sum>maxSum:
            maxSum = sum
            maxIndex = i

    selPer = permus[maxIndex]

    for k in range(numberAspects):
        recall_avg+=recallMat[k,selPer[k]]
        precision_avg+=precisionMat[k,selPer[k]]
        f1_avg+=f1Mat[k,selPer[k]]

        record_recall.append(recallMat[k,selPer[k]])
        record_precision.append(precisionMat[k,selPer[k]])
        record_f1.append(f1Mat[k,selPer[k]])

    recall_avg/=numberAspects
    precision_avg/=numberAspects
    f1_avg/=numberAspects
    trueIndex = list(selPer)

    return recall_avg, precision_avg, f1_avg, trueIndex, record_recall, record_precision, record_f1


def cal_MSE_mulGraph(networks, groundTGraphs, numberAspects, rateList, trueIndex):
    numberNodes = networks[0].node_num
    MSE = 0
    record_MSE = []
    for k in range(numberAspects):
        inferTransRate = networks[k].edge
        index = trueIndex[k]
        groundTruthTransRate = groundTGraphs[index] * rateList[index]
        cur_MSE = np.sum(np.square(inferTransRate - groundTruthTransRate))/(numberNodes * numberNodes)
        MSE += cur_MSE
        record_MSE.append(cur_MSE)

    return MSE/numberAspects, record_MSE


def cal_MAE_mulGraph(networks, groundTGraphs, numberAspects, rateList, trueIndex):
    MAE = 0
    record_MAE = []
    for k in range(numberAspects):
        inferTransRate = networks[k].edge
        index = trueIndex[k]
        groundTruthTransRate = groundTGraphs[index] * rateList[index]
        midInferTransRate = inferTransRate * groundTGraphs[index]
        tmp = groundTruthTransRate.copy()
        nonZeroCnt = np.sum(groundTGraphs[index])
        tmp[np.where(tmp==0)]=1
        cur_MAE = (np.sum(abs(midInferTransRate - groundTruthTransRate)/tmp)/nonZeroCnt)
        MAE += cur_MAE
        record_MAE.append(cur_MAE)

    return MAE/numberAspects, record_MAE



def cal_NMI(A, B):
    # 样本点数
    total = len(A)
    A_ids = set(A)
    B_ids = set(B)
    # 互信息计算
    MI = 0
    eps = 1.4e-16
    for idA in A_ids:
        for idB in B_ids:
            idAOccur = np.where(A == idA)
            idBOccur = np.where(B == idB)
            idABOccur = np.intersect1d(idAOccur, idBOccur)
            px = 1.0 * len(idAOccur[0]) / total
            py = 1.0 * len(idBOccur[0]) / total
            pxy = 1.0 * len(idABOccur) / total
            MI = MI + pxy * math.log(pxy / (px * py) + eps, 2)
    # 标准化互信息
    Hx = 0
    for idA in A_ids:
        idAOccurCount = 1.0 * len(np.where(A == idA)[0])
        Hx = Hx - (idAOccurCount / total) * math.log(idAOccurCount / total + eps, 2)
    Hy = 0
    for idB in B_ids:
        idBOccurCount = 1.0 * len(np.where(B == idB)[0])
        Hy = Hy - (idBOccurCount / total) * math.log(idBOccurCount / total + eps, 2)
    MIhat = 2.0 * MI / (Hx + Hy)
    return MIhat


def Ours(groundTGraphs, comb_cascade, cat_num, beta, rateList, debug_display=False, choice=0):
    beginTime = time.time()
    cascade_num = comb_cascade.cascade_num
    node_num = comb_cascade.node_num

    comb_cascade.init_cat_labels()
    
    # 初始 label 隔1个调正确
    # start = 0
    # while start < cascade_num:
    #     comb_cascade.labels[start] = int(start/beta)
    #     start += 2
    
    # beta ratio 初始 label 隔1个调正确
    start = 0
    while start < cascade_num:
        if start < beta[0]:
            comb_cascade.labels[start] = 0
        elif beta[0] <= start < beta[0]+beta[1]:
            comb_cascade.labels[start] = 1
        else:
            comb_cascade.labels[start] = 2
        start += 2

    piValue = np.zeros(cat_num)   
    for cat in range(cat_num):
        piValue[cat] = np.sum(comb_cascade.labels == cat) / cascade_num
    print("initial piValue: ",piValue)

    # print("init with accuracy: %.2f %%" % (100 * comb_cascade.cal_labels_accuracy()))


    ground_truth_labels = np.ones([cascade_num])
    # for i in range(int(cascade_num / 2)):
    #     ground_truth_labels[i] = 0

    # for i in range(int(cascade_num)):
    #     ground_truth_labels[i] = int(i/beta)
    
    ground_truth_labels[:beta[0]] = 0
    ground_truth_labels[beta[0]:beta[0]+beta[1]] = 1
    ground_truth_labels[beta[0]+beta[1]:beta[0]+beta[1]+beta[2]] = 2

    # comb_cascade.label_renew(ground_truth_labels)

    sub_net_list = []
    for i in range(cat_num):  # 初始化cat_num个m_network
        tmp_net = nt.m_network(node_num)
        sub_net_list.append(tmp_net)
        tmp_net = None
    theta_dict = []    # 记录每个aspect的各节点(i)各父节点组合(j)的theta值
    for i in range(cat_num):
        tmp_dict = {}
        theta_dict.append(tmp_dict)  
        tmp_dict = None
    cnt = 0
    while True:
        cnt += 1
        # first step: generate network
        # 1.1 generate network structure
        # clear all sub-graphs
        for i in range(cat_num):  # sub_net_list: 每个aspect的边
            sub_net_list[i].clear_graph()  #清空，都置为0
            sub_net_list[i] = net_util.net_parent_set_reconstruct_5a(sub_net_list[i], comb_cascade, i, 0)
            # sub_net_list[i] = net_util.net_parent_set_reconstruct_aaai(sub_net_list[i], comb_cascade, i, 0)
            # sub_net_list[i] = net_util.aaai_construct(sub_net_list[i], comb_cascade, i, 0)
            print("graph %d has %d edges " % (i, sub_net_list[i].edge_num))
        # os.system("pause")

        # 1.2 update theta
        for i in range(cat_num):
            theta_dict[i].clear()
            theta_dict[i] = net_util.get_theta(theta_dict[i], comb_cascade, i, sub_net_list[i], display=0)
            # sub_net_list[cat].display_edges()
            print("finish calculating theta for graph %d" % i)
            # os.system("pause")
        
        # import json
        # jsObj = json.dumps(theta_dict, indent=4)  # indent参数是换行和缩进
        # fileObject = open('1.json', 'w')
        # fileObject.write(jsObj) 
        # fileObject.close()  # 最终写入的json文件格式

        # second step: re-assign labels of cascades
        finish_label = True
        record_score = np.zeros(node_num)
        record_value_i = np.zeros(node_num)
        record_aspect = np.zeros(cat_num)
                    
        
        for c in range(cascade_num):
            max_likelihood = -np.inf
            max_label = -1
            for cat in range(cat_num):
                score = 1
                for i in range(node_num):
                    parent_id_set = np.argwhere(sub_net_list[cat].graph[:, i] == 1).flatten()  # 父节点集合
                    parent_state = comb_cascade.cascade[c][parent_id_set.astype(int)]  # 从这条记录中取出父节点的状态
                    # 取出theta值
                    key = ''.join(str(int(s)) for s in parent_state)
                    value = theta_dict[cat][i].get(key) 
                    if comb_cascade.cascade[c][i] == 1:
                        value_i = value
                    else:
                        value_i = 1-value
                    score *= value_i
                    record_value_i[i] = value_i
                    record_score[i] = score
                record_aspect[cat] = score
                if score > max_likelihood:
                    max_likelihood = score
                    max_label = cat
            if finish_label and max_label != comb_cascade.labels[c]:
                finish_label = False
            comb_cascade.labels[c] = max_label
            

        if debug_display:
            print("iter %d times" % cnt)
            # print(" with accuracy: %.2f %%" % (100 * comb_cascade.cal_labels_accuracy()))

            # F1, MSE, MAE, NMI
            recall, precision, f1, trueIndex, record_recall, record_precision, record_f1 = cal_F1_average_mulGraph(sub_net_list, groundTGraphs, cat_num)
            MSE, record_MSE = cal_MSE_mulGraph(sub_net_list, groundTGraphs, cat_num, rateList, trueIndex)
            MAE, record_MAE = cal_MAE_mulGraph(sub_net_list, groundTGraphs, cat_num, rateList, trueIndex)
            NMI = cal_NMI(comb_cascade.labels, ground_truth_labels)

            print("record_f1: ",record_f1)
            print("f1_std = %.5f"%(np.std(record_f1)))
            print("recall=%.3f, precision=%.3f, f1=%.3f" % (recall, precision, f1))

            print("record_MSE: ",record_MSE)
            print("MSE_std = %.5f"%(np.std(record_MSE)))
            print("MSE=%.5f" % (MSE))

            print("record_MAE: ",record_MAE)
            print("MAE_std = %.5f"%(np.std(record_MAE)))
            print("MAE=%.5f" % (MAE))

            print("NMI=%.5f" % (NMI))

            piValue = np.zeros(cat_num)
            for cat in range(cat_num):
                piValue[cat] = np.sum(comb_cascade.labels==cat)/cascade_num
            print(piValue)

            print(comb_cascade.labels)
            print("\n\n")
        else:
            print("iter %d times" % cnt)

        midTime = time.time()
        print("curTime cost: ",(midTime-beginTime)*1000)

        if finish_label:
            endTime = time.time()
            print("time cost: ",(endTime-beginTime)*1000)
            print("\n\n\nProcess finished! \n\n\n")
            break

