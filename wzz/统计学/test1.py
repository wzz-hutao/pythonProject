import pandas as pd

# # 1.2 (1)
# df = pd.read_csv("D:/资料/统计学/chap01/exercise1_2.csv",encoding='gbk')
# tab1 = pd.crosstab(df.Sex,df.Survived,margins=True,margins_name='合计')
# print(tab1)

# # 1.2 (2)
# df = pd.read_csv("D:/资料/统计学/chap01/exercise1_2.csv",encoding='gbk')
# tab = pd.pivot_table(df,index=['Survived','Sex'],
#                      columns=['Class','Age'],
#                      margins=True,margins_name='合计',
#                      aggfunc=len)
# print(tab)

# # 1.1
# s = {"收入户等级": ["低收入户","中等偏下户","中等收入户","中等偏上户","高收入户"],
#      "2016年": [3750,7338,10508,14823,28225],
#      "2017年": [4647,9330,13506,19404,36957],
#      "2018年": [6545,12674,18277,26044,49175],
#      "2019年": [8004,17024,24832,35576,67132],
#      "2020年": [10422,21636,31685,45639,85541]}
# table = pd.DataFrame(s)
# table.to_csv("D:/资料/统计学/chap01/df1",index=True,encoding='utf-8')

# import numpy.random as npr
#
# r = npr.normal(loc=200,scale=10,size=1000)
# example1_1 = pd.cut(r,bins=100)
# print(example1_1.value_counts())