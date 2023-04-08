import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn.datasets import load_iris

iris = load_iris()


# df = pd.DataFrame(iris,columns=['SepalLength','SepalWidth','PetalLength','PetalWidth'])
# df['class'] = iris['target']
# df['SepalLength'] = iris.data[:,0]
# df['SepalWidth'] = iris.data[:,1]
# df['PetalLength'] = iris.data[:,2]
# df['PetalWidth'] = iris.data[:,3]
# df.to_csv('D:/资料/数据挖掘/iris.csv',index=None)

data = pd.read_csv('D:/资料/数据挖掘/iris.csv',encoding='gbk')
# print(data)


# plt.rcParams['font.sans-serif'] = ['SimHei']
#
# plt.figure(figsize=(10,8))
# colors = ['red','blue','green']
# labels = ['setosa','versicolor','Vinginica']
#
# for i in range(3):
#     plt.scatter(x=data['SepalLength'][i*50:(i+1)*50],
#                 y=data['SepalWidth'][i*50:(i+1)*50],
#                 c=colors[i],
#                 label=labels[i])
#
#
# plt.legend()
# plt.show()

data['scale'] = data['SepalLength'] / data['SepalWidth']
print(data)

# 标准化
from sklearn.preprocessing import MinMaxScaler
scale = MinMaxScaler()
data1 = scale.fit_transform(data)
data1 = data1[:,:4]
# print(data1)

# 创建PCA对象
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca = pca.fit(data1)

# 绘制贡献度曲线
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of components')
plt.ylabel('Cumulayive explained variance')

# 根据贡献度选择合适的维度
threshold = 0.95
n_components = np.argmax(np.cumsum(pca.explained_variance_ratio_) >= threshold) + 1
print('Number of omponents to retain 95% varvance', n_components)

# 进行PCA降维
pca = PCA(n_components=n_components)
data_pca = pca.fit_transform(data1)
print(data_pca)

colors = ['red','blue','yellow']
labels = ['setosa','versicolor','Vinginica']
plt.figure()
for i in range(3):
    plt.scatter(x=data_pca[i*50:(i+1)*50, 0],
                y=data_pca[i*50:(i+1)*50, 1],
                alpha=1,
                c=colors[i],
                label=labels[i])

plt.title('Scatter Plot')
plt.show()











# movie_data = pd.read_csv("D:/python爬虫/douban.csv",encoding='gbk')
# # movie_data.dropna(axis='index',how='any',inplace=False)
# data = movie_data.loc[:,['score','num']]
#
# # print(data)
#
# from sklearn.preprocessing import StandardScaler
#
# tool = StandardScaler()
# data1 = tool.fit_transform(data)



# # axis=0  返回的是一列的均值
# mean_val = np.mean(data1,axis=0)
# # 取中心化
# mean_removed = data1 - mean_val
# # 获取协方差矩阵
# cov_mat = np.cov(mean_removed, rowvar=0)
# # 获取矩阵特征值与特征向量（计算贡献率）
# # eigen_vals特征根  eigen_vecs特征向量
# eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)
# percentage = eigen_vals/sum(eigen_vals)
# # print(percentage)
# # 特征根排序，接着删除解释量小的特征根
# eigen_val_ind = np.argsort(eigen_vals)
# # print(eigen_val_ind)
# topNfeat = 1
# eigen_val_ind = eigen_val_ind[-1:(-topNfeat+1):-1]
# # print(eigen_val_ind)
# # 特征向量由高到低排序 取最大的n个值对应的特征向量
# red_eigen_vecs = eigen_vecs[:,eigen_val_ind]
# # print(red_eigen_vecs)
# # 将原始数据投影到主成分上得到新的低维数据low_data
# # 矩阵相乘 @
# low_data = mean_removed @ red_eigen_vecs
# # 得到重构数据reconMat
# recon_mat = (low_data @ red_eigen_vecs.T) + mean_val
# print(recon_mat)
#
# # 绘图
# # plt.figure(figsize=(8,6))
# # x1 = data1[:,0]
# # y1 = data1[:,1]
# # plt.scatter(x=x1,y=y1,color='green',alpha=0.6,s=30,label='original data')
#
# x2 = data1[:,0]
# y2 = data1[:,1]
# plt.scatter(x=x2,y=y2,color='purple',alpha=0.6,s=30,label='after z-score')
#
# x3 = np.array(recon_mat)[:,0]
# y3 = np.array(recon_mat)[:,1]
# plt.scatter(x=x3,y=y3,color='yellow',alpha=0.5,s=30,label='recon_mat')
#
# plt.legend()
# plt.show()


