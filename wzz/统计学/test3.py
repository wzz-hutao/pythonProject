# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns

# # 核密度图
# df = pd.read_csv("D:/资料/统计学/chap02/example2_3.csv",encoding='gbk')
# plt.rcParams['font.sans-serif'] = ['SimHei']
# plt.rcParams['axes.unicode_minus'] = False
#
# sns.set_style('darkgrid')
# plt.subplots(1,3,figsize=(15,4))
#
# plt.subplot(131)
# sns.kdeplot(df['北京'],bw_adjust=0.2)
# plt.title('(a) bw_adjust=0.2')
#
# plt.subplot(132)
# sns.kdeplot(df['北京'],bw_adjust=0.5)
# plt.title('(b) bw_adjust=0.5')
#
# plt.subplot(133)
# sns.kdeplot(df['北京'],bw_method=0.5)
# plt.title('(c) bw_method=0.5')
#
# plt.show()


# 箱型图
# plt.rcParams['font.sans-serif'] = ['SimHei']
# # sns.set_style('darkgrid')
#
# example2_3 = pd.read_csv("D:/资料/统计学/chap02/example2_3.csv",encoding='gbk')
# # sns.set_style('darkgrid')
#
# df = pd.melt(example2_3, value_vars=['北京','上海','郑州','武汉','西安','沈阳'],
#              var_name='城市', value_name='AQI')
#
# plt.figure(figsize=(11,7))
#
# sns.boxplot(x='城市',y="AQI",
#             width=0.6,
#             saturation=0.9,
#             fliersize=2,
#             linewidth=0.8,
#             notch=False,
#             palette="Set2",
#             orient="v",
#             data=df)
#
# plt.show()


# # 小提琴图
# plt.rcParams['font.sans-serif'] = ['SimHei']
# plt.rcParams['axes.unicode_minus'] = False
# # sns.set_style('darkgrid')
#
# example2_3 = pd.read_csv("D:/资料/统计学/chap02/example2_3.csv",encoding='gbk')
# # sns.set_style('darkgrid')
#
# df = pd.melt(example2_3, value_vars=['北京','上海','郑州','武汉','西安','沈阳'],
#              var_name='城市', value_name='AQI')
#
# plt.figure(figsize=(11,7))
#
# sns.violinplot(x='城市',y="AQI",
#             width=0.6,
#             saturation=0.9,
#             fliersize=2,
#             linewidth=0.8,
#             notch=False,
#             palette="Set2",
#             orient="v",
#             inner='box',
#             data=df)
#
# plt.show()


# # 点图
# plt.rcParams['font.sans-serif'] = ['SimHei']
#
# example2_3 = pd.read_csv("D:/资料/统计学/chap02/example2_3.csv",encoding='gbk')
# df = pd.melt(example2_3, value_vars=['北京','上海','郑州','武汉','西安','沈阳'],
#              var_name='城市', value_name='AQI')
#
# plt.subplots(1,2,figsize=(10,5))
#
# plt.subplot(121)
# sns.stripplot(x='城市',y='AQI',
#               jitter=False,  # 不扰动数据
#               size=2,  # 设置点的大小
#               data=df)
# plt.title('(a) 原始数据的点图')
#
# plt.subplot(122)
# sns.stripplot(x='城市',y='AQI',
#               size=2,
#               data=df)
# plt.title('(b) 数据扰动后的点图')
#
# plt.show()



import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# 2.1(1)
data = pd.read_csv("D:/资料/统计学/chap02/exercise2_1.csv",encoding='gbk')
plt.rcParams['font.sans-serif'] = ['SimHei']

t = pd.crosstab(data.Sex,data.Survived)
t.plot(kind='bar')
plt.title('(a) 垂直条形图')
plt.xticks(rotation=0)

t1 = pd.crosstab(data.Sex,data.Survived)
t1.plot(kind='bar',stacked=True)
plt.title('(b) 垂直堆叠条形图')
plt.xticks(rotation=0)
plt.show()



plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 2.2(1)
plt.figsize = (8,6)
data1 = pd.read_csv("D:/资料/统计学/chap02/exercise2_2.csv",encoding='gbk')
sns.kdeplot(data1['eruptions'],bw_adjust=0.2)
plt.title('(a) eruptions直方图')
plt.show()


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 2.2(2)
exercise2_2_2 = pd.read_csv("D:/资料/统计学/chap02/exercise2_2.csv",encoding='gbk')
data = pd.melt(exercise2_2_2, value_vars=['eruptions','waiting'],
                var_name='类型', value_name='时间')
print(data)
sns.kdeplot('时间',hue='类型',linewidth=0.6,data=data)
plt.title("核密度比较曲线")
plt.tight_layout()

plt.show()

# 2.2(3)-箱型图
plt.subplots(1,2,figsize=(12,8))
plt.subplot(121)
exercise2_2_3 = pd.read_csv("D:/资料/统计学/chap02/exercise2_2.csv",encoding='gbk')
data = pd.melt(exercise2_2_3, value_vars=['eruptions'],
             var_name='喷发持续时间', value_name='er_time')

sns.boxplot(x='喷发持续时间',y="er_time",
            width=0.6,
            saturation=0.9,
            fliersize=2,
            linewidth=0.8,
            notch=False,
            palette="Set2",
            orient="v",
            data=data)
plt.title("eruptions_箱型图")

plt.subplot(122)
exercise2_2_4 = pd.read_csv("D:/资料/统计学/chap02/exercise2_2.csv",encoding='gbk')
data = pd.melt(exercise2_2_4, value_vars=['waiting'],
             var_name='喷发等待时间', value_name='vait_time')

sns.boxplot(x='喷发等待时间',y="vait_time",
            width=0.6,
            saturation=0.9,
            fliersize=2,
            linewidth=0.8,
            notch=False,
            palette="Set2",
            orient="v",
            data=data)
plt.title("waiting_箱型图")
plt.show()



# # 2.2(3)-小提琴图
plt.subplots(1,2,figsize=(12,8))
plt.subplot(121)
exercise2_2_5 = pd.read_csv("D:/资料/统计学/chap02/exercise2_2.csv",encoding='gbk')
df = pd.melt(exercise2_2_5, value_vars=['eruptions'],
             var_name='喷发持续时间', value_name='er_time')

sns.violinplot(x='喷发持续时间',y="er_time",
            width=0.6,
            saturation=0.9,
            fliersize=2,
            linewidth=0.8,
            notch=False,
            palette="Set2",
            orient="v",
            inner='box',
            data=df)
plt.title("eruptions_小提琴图")

plt.subplot(122)
exercise2_2_5 = pd.read_csv("D:/资料/统计学/chap02/exercise2_2.csv",encoding='gbk')
df = pd.melt(exercise2_2_5, value_vars=['waiting'],
             var_name='喷发持续时间', value_name='er_time')

sns.violinplot(x='喷发持续时间',y="er_time",
            width=0.6,
            saturation=0.9,
            fliersize=2,
            linewidth=0.8,
            notch=False,
            palette="Set2",
            orient="v",
            inner='box',
            data=df)
plt.title("waiting_小提琴图")
plt.show()
# 2.2(4) 点图
example2_2_4 = pd.read_csv("D:/资料/统计学/chap02/exercise2_2.csv",encoding='gbk')
df = pd.melt(example2_2_4, value_vars=['eruptions','waiting'],
             var_name='时间', value_name='time')

plt.subplots(1,2,figsize=(10,5))

plt.subplot(121)
sns.stripplot(x='时间',y="time",
              jitter=False,  # 不扰动数据
              size=2,  # 设置点的大小
              data=df)
plt.title('(a) eruptions&waiting点图')

plt.subplot(122)
sns.stripplot(x='时间',y="time",
              size=2,
              data=df)
plt.title('(b) 数据扰动后的eruptions&waiting点图')

plt.show()




import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
# 2.3(1)
df = pd.read_csv("D:/资料/统计学/chap02/exercise2_3.csv",encoding='gbk')

sns.jointplot(x=df['wt'],y=df['mpg'].values,marker='.',
              kind='reg',
              height=5,
              ratio=3,
              data=df)

plt.show()

# 2.3(2)
df = pd.read_csv("D:/资料/统计学/chap02/exercise2_3.csv",encoding='gbk')
sns.pairplot(df[['mpg','cyl','disp','hp','drat','wt','qsec','vs','am','gear','carb']],
             height=2,  # 子图的高度
             diag_kind='hist',  # 设置对角线的图形类型
             kind='scatter')  # 设置子图类型，默认为scatter

plt.show()







# 2.3(4)-气泡图
df = pd.read_csv("D:/资料/统计学/chap02/exercise2_3.csv",encoding='gbk')
plt.scatter(x='wt',y='hp',
            c='mpg',
            s=df['mpg']*50,
            cmap='Blues',
            edgecolors='k',
            linewidths=0.5,
            alpha=0.6,
            data=df)

plt.colorbar()
plt.xlabel('wt')
plt.ylabel('hp')
plt.title('英里数=气泡图')
plt.show()



from mpl_toolkits.mplot3d import Axes3D
# 2.3(4)-气泡图
df = pd.read_csv("D:/资料/统计学/chap02/exercise2_3.csv",encoding='gbk')
ax3d = plt.figure(figsize=(10,7)).add_subplot(111,projection='3d')
ax3d.scatter(df['wt'],df['hp'],df['mpg'],color='black',marker='*',s=50)
ax3d.set_xlabel('wt',fontsize=12)
ax3d.set_ylabel('hp',fontsize=12)
ax3d.set_zlabel('mpg',fontsize=12)
plt.xlabel('x=wt',fontsize=12)
plt.ylabel('y=hp',fontsize=12)
plt.title('z=英里数',fontsize=12)
plt.show()




import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['font.sans-serif'] = ['SimHei']

# 2.4(1)
df = pd.read_csv("D:/资料/统计学/chap02/exercise2_4.csv",encoding='gbk')

dfs = [df['Sepal.Length'],df['Sepal.Width'],df['Petal.Length'],df['Petal.Width']]
sns.lineplot(data=dfs,markers=True)
plt.xlabel("Species")
plt.ylabel('petal.length')
plt.xticks(range(150),df['Species'])

plt.show()









import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 2.4
df = pd.read_csv("D:/资料/统计学/chap02/exercise2_4.csv",encoding='gbk')
at = list(df.columns[1:4])
values_list = [[0] * 4 for i in range(3)]
for i in range(3):
    for j in range(4):
         values = df.values[i*50:(1+i)*50, j]
         values_list[i][j] = int(values.mean())

names = list(set(df.values[:,-1]))
angles = [n / float(len(values_list)) * 2 * np.pi for n in range(len(values_list))]

angles += angles[:1]
values_list = np.asarray(values_list)
values_list = np.concatenate([values_list,values_list[:,0:1]],axis=1)

plt.figure(figsize=(8,8))

for i in range(3):
    ax = plt.subplot(1,3,i+1,polar=True)
    ax.plot(angles,values_list[i][:4])
    ax.set_yticks(np.arange(0,10))
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(at)
    ax.set_title(names[i],fontsize=12,color='red')
    plt.tight_layout()

plt.show()



import pandas as pd
import matplotlib.pyplot as plt

# 2.5
plt.rcParams['font.sans-serif'] = ['SimHei']

df = pd.read_csv("D:/资料/统计学/chap02/example2_3.csv",encoding='gbk')
df = df[:31]
df['日期'] = pd.to_datetime(df['日期'])  # 将数据转化为日期格式
df = df.set_index('日期')

df.plot(kind='line',figsize=(8,5),grid=True,
        stacked=False,
        linewidth=1,
        marker='o',markersize=6,
        xlabel='年份',ylabel='AQI')

plt.title('2022年1月份AQI')
plt.show()





# 2.5
plt.rcParams['font.sans-serif'] = ['SimHei']

df = pd.read_csv("D:/资料/统计学/chap02/example2_3.csv",encoding='gbk')
df = df[:31]
df['日期'] = pd.to_datetime(df['日期'])  # 将数据转化为日期格式
df = df.set_index('日期')

df.plot(kind='area',figsize=(8,5),grid=True,
        stacked=True,
        alpha=0.5,
        xlabel='年份',ylabel='AQI')

plt.title('2022年1月份AQI')
plt.show()





















