# import pandas as pd
# import matplotlib.pyplot as plt
# import numpy as np
# df = pd.read_csv("D:/资料/数据挖掘/birth-rate.csv")
# print(df)

# # 查看数据基本情况describe()等方法应用
# print(df.shape)
# print(df.head(5))  # 前五行
# print(df.tail(5))  # 后五行
# print(df.info())
# print(df.describe())

# # 绘制某一年的直方图:
# df2008 = df['2008']
# df2008_dropna = df2008.dropna()  # 删除缺失值
# print(df2008_dropna)
# plt.hist(df['2008'],bins=np.arange(df['2008'].min(),df['2008'].max(),2),density=True,color='green',edgecolor='w' )
# plt.title("Birth-rate-2008",color='navy',fontsize=18)
# plt.xlabel("birthrate")
# plt.ylabel("country numbers")
# plt.show()


# # 箱型图和异常值分析
# data = pd.read_excel('D:/资料/数据挖掘/catering_sale.xls')
# plt.figure()
# p = data.boxplot(return_type='dict')
# if len(p['fliers']) == 0:
#     print("No outliers found in the boxplot")
# else:
#     x = x_value = []
#     y = y_value = []
#     for flier in p['fliers']:
#         x.append(flier.get_xdata())
#         y.append(flier.get_ydata())
#
# y.sort()
# x,y = x[0],y[0]
# for i in range(len(x)):
#     if i > 0:
#         plt.annotate(y[i],xy = (x[i],y[i]),xytext=(x[i]+0.06-8/(y[i]-y[i-1]),y[i]))
#     else:
#         plt.annotate(y[i],xy = (x[i],y[i]),xytext=(x[i]+0.08,y[i]))
# plt.show()


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# import xlwt
# data = pd.read_excel('D:/资料/数据挖掘/catering_sale.xls')
# sta = data.describe()
# l_fence = sta.loc['l_fence'] = sta.loc['mean'] - 3 * sta.loc['std']
# u_fence = sta.loc['u_fence'] = sta.loc['mean'] + 3 * sta.loc['std']
# data_fliers = data[(data['销量'] > int(u_fence[0])) | (data['销量'] < int(l_fence[0]))]
# print(int(l_fence[1]),int(u_fence[1]))
# list_fliers_index = data_fliers['销量'].index.tolist()
# print(data_fliers)
# print(list_fliers_index)

# data['销量1'] = data['销量']
# for i in list_fliers_index:
#     data['销量1'][i] = None
# print(data)
#
# data['销量2'] = data['销量1'].interpolate()  # 线性插值
#
# outputfile = 'D:/资料/数据挖掘/catering_sale.xls'
# data.to_excel(outputfile)