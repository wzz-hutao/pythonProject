import pandas as pd
import matplotlib.pyplot as plt

air_data = pd.read_csv("D:/资料/数据挖掘/air_data.csv")
# 1
print(air_data)
sta = air_data.describe()


# 2
print(sta.loc['min'],sta.loc['max'])
air_data_is_null = air_data.isnull().sum()
print(air_data_is_null)


# 3
plt.rcParams['font.sans-serif'] = ['SimHei']
air_data['FFP_DATE'].dropna()
date_year = dict()
for i in air_data['FFP_DATE']:
    if not date_year or i[:4] not in date_year:
        date_year[i[:4]] = 1
    date_year[i[:4]] += 1
date_year_sort = dict(sorted(date_year.items(),key=lambda x:x[0], reverse=False))

x_data = list(date_year_sort.keys())
y_data = list(date_year_sort.values())

plt.bar(x_data,y_data)
plt.xlabel("年份")
plt.ylabel("入会人数")
plt.title("各年份会员入会人数")
plt.show()


# 4
air_data = pd.read_csv("D:/资料/数据挖掘/air_data.csv")
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.figure(figsize=(12,8))
colors = ['slateblue','lawngreen']
air_data_gender = {'男':0,'女':0}
for i in air_data['GENDER']:
    if i == '男':
        air_data_gender['男'] += 1
    else:
        air_data_gender['女'] += 1

p2 = plt.pie(air_data_gender.values(), labels=air_data_gender.keys(),# 扇形提供标签的字符串序列
             autopct='%1.1f%%',  # 小数设置
             radius=1.2,  # 饼半径
             pctdistance=0.5,  # 数据的半径
             colors=colors,)
             # wedgeprops=dict(linewidth=1.2,width=0.3,edgecolor='w'))  # 边缘 线宽 环宽
plt.title('会员性别比例')
plt.show()


# 5
air_data = pd.read_csv("D:/资料/数据挖掘/air_data.csv")
plt.rcParams['font.sans-serif'] = ['SimHei']
air_data_MEMBER_NO = {4:0,5:0,6:0}
for i in air_data['FFP_TIER']:
    air_data_MEMBER_NO[i] += 1

x_data = list(air_data_MEMBER_NO.keys())
y_data = list(air_data_MEMBER_NO.values())

plt.bar(x_data,y_data)
plt.title('会员各级别人数')
plt.xlabel('会员等级')
plt.ylabel('会员人数')
plt.show()