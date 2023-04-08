# bs4解析-html语法
# <标签 属性="属性值">被标记的内容</标签>
# <a href="http://www.baidu.com">周杰伦</a>
# 例外：<img src="xxx.jpg"/> 图片        <br />


import requests
from bs4 import BeautifulSoup
import csv


# # 北京新发地爬取示例，这个地址已经不存在了
# # 获取页面源代码
# # 使用bs4进行解析，拿到数据
#
# url = "http://www.xinfadi.com.cn/marketanalysis/0/list/1.shtml"
# resp = requests.get(url)
# # print(resp.text)
# f = open("北京新发地菜价.csv",mode="w")
# csvwriter = csv.writer(f)
#
# # 解析数据
# # 把页面源代码交给BeautifulSoup进行处理，生产bs对象
# # html.parser 指定html解析器
# page = BeautifulSoup(resp.text,"html.parser")
# # 从bs对象中查找数据
# # find(标签名，属性=属性值)：找一个数据返回
# # find_all(标签名，属性=属性值)：找一堆数据返回
# # class是python的一个关键字，直接写会报错
# # table = page.find("table",class_="hq_table")
# # 和上一行是一个意思，这样书写可以避免class
# table = page.find("table",attrs={"class":"hq_table"})
# print(table)
# 拿到所有的数据
# trs = table.find_all("tr")[1:]
# for tr in trs:
#   tds = tr.find_all("td")
#   name = tds[0].text #.text 表示拿到被标签记的内容
#   low = tds[1].text #.text 表示拿到被标签记的内容
#   avg = tds[2].text #.text 表示拿到被标签记的内容
#   high = tds[3].text #.text 表示拿到被标签记的内容
#   gui = tds[4].text #.text 表示拿到被标签记的内容
#   kind = tds[5].text #.text 表示拿到被标签记的内容
#   date = tds[6].text #.text 表示拿到被标签记的内容
#   print(name,low,avg,high,gui,kind,date)
#   csvwriter.writerow([name,low,avg,high,gui,kind,date])
#
# f.close()






