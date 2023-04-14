import pandas as pd
import numpy as np

example3_1 = pd.read_csv("D:/资料/统计学/数据/例题数据/pydata/chap03/example3_1.csv",encoding='gbk')
print('分数平均值：',example3_1['分数'].mean())

example3_2 = pd.read_csv("D:/资料/统计学/数据/例题数据/pydata/chap03/example3_2.csv",encoding='gbk')
m = example3_2['组中值']
f = example3_2['人数']
print("加权平均：",np.average(a=m,weights=f))

print("中位数：",example3_1['分数'].median())

print("四分位数：",pd.DataFrame.quantile(example3_1,q=[0.25,0.75],interpolation='linear'))

print("百分位数：",pd.DataFrame.quantile(example3_1,q=[0.1,0.25,0.5,0.75,0.9],interpolation='linear'))

print("众数：",pd.DataFrame.mode(example3_1))

print("极差：",example3_1['分数'].max() - example3_1['分数'].min())

print("四分位差：",np.quantile(example3_1['分数'],q=[0.75]) - np.quantile(example3_1['分数'],q=[0.25]))

print("方差：",example3_1['分数'].var())

print("标准差：",round(example3_1['分数'].std(),2))


example2_3 = pd.read_csv("D:/资料/统计学/数据/例题数据/pydata/chap02/example2_3.csv",encoding='gbk')
s_mean = example2_3.mean(numeric_only=True)
s_sd = example2_3.std(numeric_only=True)
s_cv = s_sd / s_mean
df = pd.DataFrame({"平均数": s_mean, "标准差": s_sd, "变异系数": s_cv})
print(np.round(df, 4))


from scipy import stats
z = stats.zscore(example3_1["分数"], ddof=1)
z = np.round(z, 4)
print("标准分数:", '\n',z)

skew = example3_1['分数'].skew()
kurt = example3_1['分数'].kurt()
print("偏度系数：", round(skew, 4),'\n'"峰度系数：", round(kurt, 4))


# import seaborn as sns
# import matplotlib.pyplot as plt
#
# plt.rcParams['font.sans-serif'] = ['SimHei']
#
# example3_12 = pd.read_csv("D:/资料/统计学/数据/例题数据/pydata/chap03/example3_12.csv",encoding='gbk')
# x = example3_12['月生活费支出']
# fig = plt.figure(figsize=(6,5))
# spec = fig.add_gridspec(nrows=2, ncols=1, height_ratios=[1,3])

