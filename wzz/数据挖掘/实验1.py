import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# df = pd.read_csv("D:/资料/数据挖掘/titanic_train.csv")
#
# print(df.describe())
#
# print(df.head(10))
#
# Age_is_null_len = len(df['Age'])-df['Age'].isnull().sum()
# print(Age_is_null_len)
#
# print(df['Age'].mean())


data = pd.read_csv("D:/资料/数据挖掘/titanic_train.csv")
data = data[['Age','Fare']]
plt.figure()
data.boxplot(sym="r*",vert=False,patch_artist=True,meanline=False,showmeans=True)
plt.show()