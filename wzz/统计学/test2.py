import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("D:/资料/统计学/chap01/example2_2(1).csv",encoding='gbk')
plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文

# 设置画布
plt.figure(figsize=(12,8))
# plt.subplots(1,2,figsize=(12,8))
# plt.subplot(1,2,2) # 1 * 2的区间 放在第二个位置

colors = ['slateblue','lawngreen','magenta','green','orange','cyan','pink','gold']
p2 = plt.pie(df['北京'],labels=df['支出项目'],  # 扇形提供标签的字符串序列
             autopct='%1.2f%%',  # 小数设置
             radius=1.4,  # 饼半径
             pctdistance=0.92,  # 数据的半径
             colors=colors,
             wedgeprops=dict(linewidth=1.2,width=0.3,edgecolor='w'))  # 边缘 线宽 环宽

p2 = plt.pie(df['上海'],autopct='%1.2f%%',
             radius=1.1,pctdistance=0.87,
             colors=colors,
             wedgeprops=dict(linewidth=1.2,width=0.3,edgecolor='w'))

p2 = plt.pie(df['天津'],autopct='%1.2f%%',
             radius=0.8,pctdistance=0.8,
             colors=colors,
             wedgeprops=dict(linewidth=1.2,width=0.3,edgecolor='w'))

p2 = plt.pie(df['重庆'],autopct='%1.2f%%',
             radius=0.5,pctdistance=0.65,
             colors=colors,
             wedgeprops=dict(linewidth=1.2,width=0.3,edgecolor='w'))


plt.title('(a)北京，上海，天津，重庆各项消费支出的环形图',x=0.1,y=1)
plt.show()

df = pd.read_csv("D:/资料/统计学/chap01/example1_1.csv",encoding='gbk')
plt.rcParams['font.sans-serif'] = ['SimHei']
t = pd.crosstab(df.性别,df.态度)
t.plot(kind='bar')
plt.xticks(rotation=0)  # 默认rotation = 90
plt.show()



