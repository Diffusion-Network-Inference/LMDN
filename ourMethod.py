from exp import *

# import os
# print(os.path.abspath('.'))

cat_num = 3
# beta = 200
beta = [100, 200, 300]
rateList = [0.3, 0.3, 0.3, 0.3, 0.3]
numberNodes = 378
# cascade_num = cat_num * beta
cascade_num = sum(beta)
choice = 3   # choice: 0: 1-MI/sigmaMI*p   1: 1-p    2: 1-p/sigmap   3: 1-ßßp/sigmap*p   4: 1-MI/sigmaMI    5: (1-p)*MI/sigmaMI

# 200个节点
graphPathList = [r'./data/real/subgraph_0_0.15_200_0.3.txt',
                  r'./data/real/subgraph_1_0.15_200_0.3.txt',
                  r'./data/real/subgraph_2_0.15_200_0.3.txt']

# 记录条数
# base_addr = r"./data/real/"
# cascade_name = r"record_states_subgraphs012_0.15.txt"
base_addr = r"./data/add_experiment/"
cascade_name = r"record_states_subgraphs012_0.15_100_200_300.txt"

# graphPathList = [r'./data/real/DUNF_random_0.6667_0_0.15_200_0.3.txt',
#                   r'./data/real/DUNF_random_0.6667_1_0.15_200_0.3.txt',
#                   r'./data/real/DUNF_random_0.6667_2_0.15_200_0.3.txt']

# base_addr = r"./data/real/"
# cascade_name = r"record_states_DUNF_random_0.6667_0.15.txt"
# base_addr = r"./data/add_experiment/"
# cascade_name = r"record_states_DUNF_200_400_600.txt"
#
#
# print("choice = ", choice)

# graphPathList = [r'./data/syntheticData/network_200_2_0.15_200_0.3.txt',
#                   r'./data/syntheticData/network_200_3_0.15_200_0.3.txt',
#                   r'./data/syntheticData/network_200_4_0.15_200_0.3.txt',
#                   r'./data/syntheticData/network_200_5_0.15_200_0.3.txt',
#                   r'./data/syntheticData/network_200_6_0.15_200_0.3.txt']
# graphPathList = [r'./data/syntheticData/network_200_3_0.15_200_0.3.txt',
#                   r'./data/syntheticData/network_200_4_0.15_200_0.3.txt',
#                   r'./data/syntheticData/network_200_5_0.15_200_0.3.txt']
  
# base_addr = r"./data/syntheticData/"
# cascade_name = r"record_states_200_2_3_4_5_6_200.txt"
# cascade_name = r"record_states_200_3_4_5_200.txt"

print("choice = ", choice)

print(graphPathList)

groundTGraphs, comb_cascade = load_data_K(graphPathList, base_addr, cascade_name, numberNodes, cat_num,
                                          cascade_num, beta)


Ours(groundTGraphs, comb_cascade, cat_num, beta, rateList, True, choice)