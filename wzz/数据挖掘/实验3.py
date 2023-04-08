
# data = pd.read_csv("D:/资料/数据挖掘/air_data.csv",header=0,index_col=0)
# # 1.1 去除票价为空的记录
# data_notnull = data.loc[data['SUM_YR_1'].notnull() & data['SUM_YR_2'].notnull(),:]
#
# # 1.2-3
# index = (data_notnull['SUM_YR_1'] != 0) | (data_notnull['SUM_YR_2'] != 0)
# index1 = (data_notnull['avg_discount'] != 0) & (data_notnull['SEG_KM_SUM'] > 0)
# index2 = data_notnull['AGE'] > 100
#
# airline = data_notnull[index & index1 & ~index2]
# # print(airline)
# airline.to_csv("D:/资料/数据挖掘/air_data_clean.csv")
#
# # 2
# data_clean = pd.read_csv("D:/资料/数据挖掘/air_data_clean.csv")
# data_clean_s6 = data_clean[['FFP_DATE','LOAD_TIME','LAST_TO_END','FLIGHT_COUNT','SEG_KM_SUM','avg_discount']]
# # print(data_clean_s6)
#
# # 3
# L = pd.to_datetime(data_clean_s6['LOAD_TIME']) - pd.to_datetime(data_clean_s6['FFP_DATE'])
# L = L.astype('str').str.split().str[0]
# L = L.astype('int') / 30
# R = data_clean_s6['LAST_TO_END'] / 30
#
# data_change = pd.concat([L,R,data_clean_s6.iloc[:,3:]],axis=1)
# data_change.columns = ['L','R','F','M','C']
# # print(data_change)
#
# # 标准化
# from sklearn.preprocessing import StandardScaler
# tool = StandardScaler()
# data_scale = tool.fit_transform(data_change)
# # print(data_fit_c)

# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.cluster import KMeans
#
# # 等频离散化
# df = pd.read_excel("D:/资料/数据挖掘/discretization_data.xls")
# data = df[u'肝气郁结证型系数'].copy()
# k = 4
# tool = KMeans(n_clusters=k)
# tool.fit(data.values.reshape(len(data),1))
#
#
# w = [1.0*i/k for i in range(k+1)]
# w = data.describe(percentiles=w)[4:4+k+1]
# w[0] = w[0]*(1-1e-10)
# d2 = pd.cut(data, w, labels=range(k))
#
#
# def cluster_plot(d,k):
#     plt.figure(figsize=(8,3))
#     for j in range(0,4):
#         plt.plot(data[d == j], [j for i in d[d == j]], 'o')
#     plt.ylim(-0.5,k - 0.5)
#     plt.show()
#     return plt
#
# cluster_plot(d2,4)
#
# # 等宽离散化
# data = df[u'肝气郁结证型系数'].copy()
# tool = KMeans(n_clusters=4)
# tool.fit(data.values.reshape(len(data),1))
#
# centers = pd.DataFrame(tool.cluster_centers_).sort_values(0)
# borders = centers.rolling(2).mean()
# borders = borders.dropna()
# # print(borders)
# borders = [0] + list(borders[0]) + [data.max()]
#
# d3 = pd.cut(data, bins=borders,labels=range(4))
# def cluster_plot(d,k):
#     plt.figure(figsize=(8,3))
#     for j in range(0,4):
#         plt.plot(data[d == j], [j for i in d[d == j]], 'o')
#     plt.ylim(-0.5,k - 0.5)
#     plt.show()
#     return plt
#
# cluster_plot(d3,4)


# import pandas as pd
#
# data = pd.read_excel("D:/资料/数据挖掘/normalization_data.xls",header=None)  # 没有表头时候用
# # data1 = (data - data.min()) / (data.max() - data)
# from sklearn.preprocessing import MinMaxScaler
# tool = MinMaxScaler()
# data2 = tool.fit_transform(data)
# data3 = pd.DataFrame(data2)
# print(data3)

import pandas as pd
import numpy as np
data = pd.read_excel("D:/资料/数据挖掘/normalization_data.xls",header=None)
data1 = data / 10 ** np.ceil(np.log10(data.abs().max()))
print(data1)