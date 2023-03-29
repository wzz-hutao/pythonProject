from urllib.request import urlopen
import csv
import re
import requests
import json


url = 'http://www.baidu.com'
resp = urlopen(url)

with open('D:/python爬虫/baidu.html',mode='wb') as f:
    f.write(resp.read())
print("over")



# http/https协议
# 请求头最常见的重要内容
# 1.User-Agent: 请求载体的身份标识（用啥发送的请求）
# 2.Referer: 防盗链（这次请求是从哪个页面来的？反爬会用到）
# 3.cookie: 本地字符串数据信息（用户登录信息，反爬的token）
#
# 响应头中一些重要内容
# 1.cookie: 本地字符串数据信息（用户登录信息，反爬的token）
#
# 请求方式
# 1.GET: 查询
# 2.POST: 增加，修改



# pip清华源
# pip install -i https://pypi.tuna.tsinghua.edu.cn/simple + (例子requests)



# 爬取搜狗首页的页面数据
# 指定url
url = 'https://www.sogou.com/'
# 发起请求
response = requests.get(url)
# 响应对象
page_text = response.text
# 持久化储存
with open('D:/python爬虫/sogou.html','w',encoding='utf-8') as fp:
    fp.write(page_text)
print("爬取结束")
response.close()



# 搜狗某一搜索
# UA : User-Agent (请求载体的身份标识)
# UA检测 : 门户网站的服务器会检测对应请求的载体身份标识,如果检测到请求的载体身份为某一浏览器
# 说明是正常的请求，不正常的请求可能会失败
# UA伪装: 让爬虫对应的请求载体身份标识伪装成某一浏览器

# UA伪装 :将对应的User-Agent封装到一个字典中
headers = {
    'User-Agent':"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/111.0.0.0 Safari/537.36 Edg/111.0.1661.51"
}
url = 'https://www.sogou.com/web'
# 处理url的参数: 封装到字典
kw = input('enter a word:')
param = {
    'query':kw
}
# 对指定的url发起请求
response = requests.get(url,params=param,headers=headers)
page_text = response.text
filename = 'D:/python爬虫/'+kw+'.html'
with open(filename,'w',encoding='utf-8') as fp:
    fp.write(page_text)
print(filename,"保存成功")
response.close()



# 破解百度翻译
# ajax技术可以实现动态页面局部更新，文件类型一般为xhr or fetch
# post请求(携带了参数)
post_url = 'https://fanyi.baidu.com/sug'  # 'https://fanyi.baidu.com/langdetect'
word = input('enter a word:')
data = {'kw':word}
headers = {'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/111.0.0.0 Safari/537.36 Edg/111.0.1661.51'}
response = requests.post(url=post_url,data=data,headers=headers)
# Content-Type: application/json
dic_json = response.json()  # json()返回一个字典对象
filename = 'D:/python爬虫/baidu_' + word + '.json'
fp = open(filename,'w',encoding='utf-8')
json.dump(dic_json,fp=fp,ensure_ascii=False)  # 中文不用ascii
print("over")
response.close()



# 豆瓣电影排行
# baidu搜索json 在线格式化校验
url = 'https://movie.douban.com/j/chart/top_list'
headers = {'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/111.0.0.0 Safari/537.36 Edg/111.0.1661.51'}
param = {
    'type': '24',
    'interval_id': '100:90',
    'action':'',
    'start': '0',  # 从库中的第几部电影去取
    'limit': '20'   # 一次取出的个数
}
response = requests.get(url=url,params=param,headers=headers)
list_data = response.json()
filename = 'D:/python爬虫/douban.json'
fp = open(filename,'w',encoding='utf-8')
json.dump(list_data,fp=fp,ensure_ascii=False)
print("over")
response.close()




# 肯德基餐厅查询 http://www.kfc.com.cn/kfccda/index.aspx 中指定的餐厅数量
post_url = 'http://www.kfc.com.cn/kfccda/index.aspx'
headers = {'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/111.0.0.0 Safari/537.36 Edg/111.0.1661.51'}
word = input('enter a word:')
data = {'kw':word}
response = requests.post(url=post_url,data=data,headers=headers)
# Content-Type: application/json
page_text = response.text
filename = 'D:/python爬虫/kfc_' + word + '.html'
with open(filename,'w',encoding='utf-8') as fp:
    fp.write(page_text)
print(filename,"保存成功")
response.close()



# 数据解析 -正则(re)  -bs4  -xpath
# 数据解析原理概述:
# - 解析的局部的文本内容都会在标签之间或者标签对应的属性中储存
# - 1.进行指定标签的定位
# - 2.标签或者标签对应的属性中储存的数据值进行提取（解析）



# 豆瓣
url = 'https://movie.douban.com/top250'
headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/111.0.0.0 Safari/537.36 Edg/111.0.1661.54"}
resp = requests.get(url=url,headers=headers)
page_content = resp.text
# print(page_content)
obj = re.compile(r'<li>.*?<div class="item">.*?<span class="title">(?P<movie_name>.*?)'
                 r'</span>.*?<br>(?P<year>.*?)&nbsp;/&nbsp;(?P<country>.*?)&nbsp;/&nbsp;(?P<type>.*?)</p>'
                 r'.*?"v:average">(?P<score>.*?)</span>.*?<span>(?P<num>.*?)人评价</span>',re.S)
results = obj.finditer(page_content)
f = open('D:/python爬虫/douban.csv','w')
headRow = ["movie_name","year","country","type","score","num"]
csv_write = csv.writer(f)
csv_write.writerow(headRow)
for i in results:
    # print(f'name = {i.group("movie_name")}')
    # print(i.group("year").strip())
    # print(i.group("country"))
    # print(i.group("type").strip())
    # print(f'评价的人数为：{i.group("num")}')
    # print(i.group("score"),'\n')
    dic = i.groupdict()
    dic['year'] = dic['year'].strip()
    dic['type'] = dic['type'].strip()
    csv_write.writerow(dic.values())
f.close()
print("over")



# 屠戮盗版天堂电影信息
# html中a标签表示超链接 例如: <a href="url">周杰伦</a>
# <a href="https://www.dy2018.com/html/gndy/dyzz/index.html">2023新片精品</a>
# 1.定位
url = 'https://www.dytt89.com/'
resp = requests.get(url,verify=False)  # verify=False 去掉安全验证
resp.encoding = "gb2312"  # charset=gb2312 gbk也行
# print(resp.text)

obj1 = re.compile(r"2023新片精品.*?<ul>(?P<ul>.*?)</ul>",re.S)
obj2 = re.compile(r"<a href='(?P<href>.*?)'",re.S)
obj3 = re.compile(r'◎片　　名(?P<movie>.*?)<br />.*?<td style="WORD-WRAP: break-word" bgcolor="#fdfddf"><a href='
                  r'"(?P<download>.*?)">',re.S)

# 从2023必看片中提取子页面的链接地址 - htef里面的东西
results1 = obj1.finditer(resp.text)
child_url_list = []
for it in results1:
    ul = it.group("ul")
    # print(ul)
    results2 = obj2.finditer(ul)
    for itt in results2:
        # 拼接子页面的url地址
        child_url = url + itt.group("href").strip('/')
        child_url_list.append(child_url)  # 保存子页面链接

# 提取子页面内容
for href in child_url_list:
    child_resp = requests.get(href,verify=False)
    child_resp.encoding = 'gb2312'
    # print(child_resp.text)
    results3 = obj3.search(child_resp.text)
    print(results3.group("movie"))
    print(results3.group("download"))



# bs4解析-html语法
# <标签 属性="属性值">被标记的内容</标签>
# <a href="http://www.baidu.com">周杰伦</a>
# 例外：<img src="xxx.jpg"/> 图片        <br />



# 示例，这个地址已经不存在了
# 获取页面源代码
# 使用bs4进行解析，拿到数据
import requests
from bs4 import BeautifulSoup
import csv

url = "http://www.xinfadi.com.cn/marketanalysis/0/list/1.shtml"
resp = requests.get(url)
# print(resp.text)
f = open("北京新发地菜价.csv",mode="w")
csvwriter = csv.writer(f)

# 解析数据
# 把页面源代码交给BeautifulSoup进行处理，生产bs对象
# html.parser 指定html解析器
page = BeautifulSoup(resp.text,"html.parser")
# 从bs对象中查找数据
# find(标签名，属性=属性值)：找一个数据返回
# find_all(标签名，属性=属性值)：找一堆数据返回
# class是python的一个关键字，直接写会报错
# table = page.find("table",class_="hq_table")
# 和上一行是一个意思，这样书写可以避免class
table = page.find("table",attrs={"class":"hq_table"})
# print(table)
# 拿到所有的数据
trs = table.find_all("tr")[1:]
for tr in trs:
  tds = tr.find_all("td")
  name = tds[0].text #.text 表示拿到被标签记的内容
  low = tds[1].text #.text 表示拿到被标签记的内容
  avg = tds[2].text #.text 表示拿到被标签记的内容
  high = tds[3].text #.text 表示拿到被标签记的内容
  gui = tds[4].text #.text 表示拿到被标签记的内容
  kind = tds[5].text #.text 表示拿到被标签记的内容
  date = tds[6].text #.text 表示拿到被标签记的内容
  print(name,low,avg,high,gui,kind,date)
  csvwriter.writerow([name,low,avg,high,gui,kind,date])

f.close()



# 新地址处理
import requests
import json
import jsonpath
from concurrent.futures import ThreadPoolExecutor

print("名称\t最高价\t平均价\t最低价")
f = open("D:/python爬虫/北京新发地菜价.txt", "w")


def main(current):
    url = "http://www.xinfadi.com.cn/getPriceData.html"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/86.0.4240.198 Safari/537.36"}
    data = {"limit": "20",
            "current": f"{current + 1}",
            "pubDateStartTime": "",
            "pubDateEndTime": "",
            "prodPcatid": "",
            "prodCatid": "",
            "prodName": ""}
    parsePrint(url, headers, data)


def parsePrint(url, headers, data):
    res = requests.post(url, headers=headers, data=data)
    jsonbj = json.loads(res.text)
    # 解析数据
    name = jsonpath.jsonpath(jsonbj, "$..prodName")
    highPrice = jsonpath.jsonpath(jsonbj, "$..highPrice")
    avgPrice = jsonpath.jsonpath(jsonbj, "$..avgPrice")
    lowPrice = jsonpath.jsonpath(jsonbj, "$..lowPrice")
    # 打印信息
    for g in range(0, 20):
        # 输出数据
        print(f"{name[g]}\t{highPrice[g]}\t{avgPrice[g]}\t{lowPrice[g]}\n")
        # 写入数据
        f.write(f"{name[g]}\t{highPrice[g]}\t{avgPrice[g]}\t{lowPrice[g]}\n")


if __name__ == '__main__':
    # 创立线程池
    with ThreadPoolExecutor(2) as t:
        for i in range(2):  # 爬取页数
            # 给子线程提交任务并执行
            t.submit(main, current=i)
    print("下载完成")
    f.close()


