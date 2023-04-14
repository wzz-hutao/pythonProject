# bs4解析-html语法
# <标签 属性="属性值">被标记的内容</标签>
# <a href="http://www.baidu.com">周杰伦</a>
# 例外：<img src="xxx.jpg"/> 图片        <br />
import time

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





# # Bs4抓取优美图库图片
# # 1.拿到主页面的源代码，然后提取到子页面的链接地址， href
# # 2.通过href拿到子页面的内容，从子页面中找到图片的下载地址 img -> src
# # 3.下载图片
# from bs4 import BeautifulSoup
# import time
#
# url = "https://www.umei.cc/bizhitupian/weimeibizhi/"
# resp = requests.get(url)
# resp.encoding = 'utf-8'
# # print(resp.text)
# main_page = BeautifulSoup(resp.text, "html.parser")
# alist = main_page.find("div", class_="Clbc_Game_r b-1").find_all("a")
#
# # print(alist)
# s = "https://www.umei.cc"
# for a in alist:
#     href = a.get('href')
#     href = s + href
#     # print(href)
#     # 拿到子页面的源代码
#     child_page_resp = requests.get(href)
#     child_page_resp.encoding = 'utf-8'
#     child_page_text = child_page_resp.text
#     # 从子页面拿到图片的下载途径
#     child_page = BeautifulSoup(child_page_text, "html.parser")
#     divs = child_page.find("div", class_="big-pic")
#     img = divs.find("img")
#     src = img.get("src")
#     # 下载图片
#     img_resp = requests.get(src)
#     img_name = src.split("/")[-1]
#     with open('img/'+img_name, mode='wb') as f:
#         f.write(img_resp.content)
#     print("over!",img_name)
#     time.sleep(1)
# print("all over")





# # xpath 是XML文档中搜索内容的一门语言
# # html是xml的一个子集
#
# # xpath解析
# from lxml import etree
# xml = ''
# tree = etree.XML(xml)
# # text()拿文本
# # //拿b类以下所有的c子类（后代）
# # *任意的结点->通配符
# #  a[@href='abc'] = abc   <a href='abc'>abc<
# #  'abc' = a/@href
# result = tree.xpath("a/b//c/text()")
# result = tree.xpath("a/b/c[1]/text()")  # 第一个[1]
# # 相对查找用./d/text()
# Ctrl + F 搜索

# import requests
# from lxml import etree
#
# url = "https://beijing.zbj.com/search/service/?kw=saas&r=1"
# resp = requests.get(url)
# # print(resp.text)
# html = etree.HTML(resp.text)
#
# # 拿到每一个服务商的div
#
# divs = html.xpath('//*[@id="__layout"]/div/div[3]/div/div[4]/div/div[2]/div[1]/div')
# # 完整地址毛都搜不到
# # divs = html.xpath("/html/body/div[2]/div/div/div[3]/div/div[4]/div/div[2]/div[1]/div")
#
# for div in divs:
#     price = div.xpath("./div/div[2]/div[1]/span/text()")[0].strip("￥")
#     title = "saas".join(div.xpath("./div/div[2]/div[2]/a/text()"))
#     print(price)
#     print(title)





# # 处理cookie登录小说网
# # 1.登录 -> 得到cookie
# # 2.带着cookie 去请求到书架url -> 书架上的内容
# # session 一连串的请求，在这个过程的cookie不会丢失
# import requests
#
# # 会话
# session = requests.session()
# data = {
#     'loginName': '18973988107',
#     'password': 'Wzz20031123'
# }
#
# # 1.登录
# url = "https://passport.17k.com/ck/user/login"
# resp = session.post(url,data=data)
# # print(resp.text)
# # print(resp.cookies)
#
# # 2.那书架的数据
# resp_session = session.get("https://user.17k.com/ck/author/shelf?page=1&appKey=2406394919")
# print(resp_session.json())





# # 防盗链的处理
# import requests
# # 1.拿到contId
# # 2.拿到videoStatus返回的json. -> srcURL
# # 3.srcURL里面的内容进行修整
# # 4.下载视频
# url = "https://www.pearvideo.com/video_1693168"
# contId = url.split("_")[1]
#
# videoStatusUrl = f"https://www.pearvideo.com/videoStatus.jsp?contId={contId}&mrd=0.656549338823563"
# headers = {
#     "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/112.0.0.0 Safari/537.36 Edg/112.0.1722.39",
#     # 防盗链 溯源，当前请求的上一级是谁
#     "Referer": url
# }
# resp = requests.get(videoStatusUrl, headers=headers)
# dic = resp.json()
# srcUrl = dic['videoInfo']['videos']['srcUrl']
# systemTime = dic['systemTime']
# srcUrl = srcUrl.replace(systemTime,'cont-'+contId)
#
# with open("D:/python爬虫/li.mp4",mode='wb') as f:
#     f.write(requests.get(srcUrl).content)



# 代理：通过第三方的一个机器去发送请求



# # # 综合训练-抓取网易云音乐热评
# # # 1.找到未加密的参数                       # window.asrsea(参数，x,x,x,...)
# # # 2.想办法把参数进行加密（必须参照网易的逻辑）,params =>encText, encSecKey => encSecKey
# # # 3.请求到网易，拿到评论信息
# #
# #
# from Crypto.Cipher import AES  # 在python3.8中运行
# from base64 import b64encode
# import requests
# import json
#
# url = "https://music.163.com/weapi/comment/resource/comments/get?csrf_token="
#
# # POST
# data = {
#     'csrf_token': "",
#     'cursor': '-1',
#     'offset': '0',
#     'orderType': '1',
#     'pageNo': '1',
#     'pageSize': '20',
#     'rid': "R_SO_4_65800",
#     'threadId': "R_SO_4_65800"
# }
#
# # 处理加密过程
#
# 'window.asrsea(JSON.stringify(i0x), bsg8Y(["流泪", "强"]), bsg8Y(TH5M.md), bsg8Y(["爱心", "女孩", "惊恐", "大笑"]))'
#
# """
# !function() {
#     function a(a = 16) { # 随机的16位字符串
#         var d, e, b = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789", c = "";
#         for (d = 0; a > d; d += 1) # 循环16次
#             e = Math.random() * b.length, # 随机数
#             e = Math.floor(e), # 取整
#             c += b.charAt(e); # 取字符串的xxx位置
#         return c
#     }
#     function b(a, b) { # a是要加密的内容
#         var c = CryptoJS.enc.Utf8.parse(b) # b是密钥
#           , d = CryptoJS.enc.Utf8.parse("0102030405060708")
#           , e = CryptoJS.enc.Utf8.parse(a) # e是数据
#           , f = CryptoJS.AES.encrypt(e, c, { # c是加密的密钥
#             iv: d, # 偏移量
#             mode: CryptoJS.mode.CBC # 模式：cbc
#         });
#         return f.toString()
#     }
#     function c(a, b, c) { # c里面不产生随机数
#         var d, e;
#         return setMaxDigits(131),
#         d = new RSAKeyPair(b,"",c),
#         e = encryptedString(d, a)
#     }
#     function d(d, e, f, g) {
#         var h = {} # 空对象
#           , i = a(16); # 16位随机值 ，把i设置成定值
#         return h.encText = b(d, g), # g是密钥
#         h.encText = b(h.encText, i), # 返回的就是params # i就是密钥
#         h.encSecKey = c(i, e, f), # 得到的就是encSecKey ，e和f是定死的，由i决定 ，i固定，得到的key也固定
#         h
#     }
#     function e(a, b, d, e) {
#         var f = {};
#         return f.encText = c(a + e, b, d),
#         f
#     }
#     window.asrsea = d,
#     window.ecnonasr = e
#
#     两次加密：
#     数据+g => b => 第一次加密+i => b = params
# }();
# """
#
# d = data
# e = '010001'
# f = '00e0b509f6259df8642dbc35662901477df22677ec152b5ff68ace615bb7b725152b3ab17a876aea8a5aa76d2e417629ec4ee341f56135fccf695280104e0312ecbda92557c93870114af6c9d05c4f7f0c3685b7a46bee255932575cce10b424d813cfe4875d3e82047b97ddef52741d546b8e289dc6935b3ece0462db0a22b8e7'
# g = '0CoJUm6Qyw8W8jud'
# i = "Ng1djQvlJZYp2MRl"
#
# def to_16(data):
#     pad = 16 - len(data) % 16
#     data += chr(pad) * pad
#     return data
#
# def get_encSecKey():
#     return "217fc520364aa3c226288e45c451557c948e984c67b382480a7a63dbfa56973d39a00dc13109dd702e325d644e2a0771183355a8fc574c3f8bdb392d7601965d81b4d7f693adab5b3a05750a5962f8338a084524caa1ce33e90a3f9aa844f7f9ed73fe1907f7a1ff744f51ee4b0f0723758b5e51679e68f6192dca4cea924720"
#
# # 把参数进行加密
# def get_params(data):  # 默认收到的是字符串
#     first = enc_params(data, g)
#     second = enc_params(first, i)
#     return second
#
#
# def enc_params(data, key):  # 加密过程 # 默认收到的是字符串
#     iv = "0102030405060708"
#     data = to_16(data)
#     aes = AES.new(key=key.encode('utf-8'), IV=iv.encode('utf-8'), mode=AES.MODE_CBC)  # 创建加密器
#     bs = aes.encrypt(data.encode('utf-8'))  # 加密 长度必须是16的倍数
#     return str(b64encode(bs), 'utf-8')
#
#
# resp = requests.post(url, data={
#     'params': get_params(json.dumps(data)),
#     "encSecKey": get_encSecKey()
# })
#
#
# str = resp.content.decode()
# dic_json = json.loads(str)
#
# hotComments= dic_json["data"]["hotComments"]
#
# with open("D:/python爬虫/hotcomments.txt","w",encoding='utf-8') as file:
#     for i in hotComments:
#         file.writelines(i["content"]+"\n")
#     print("over!")
#
# resp.close()
# file.close()





# 多线程
# 进程是资源单位，每个进程至少由一个线程
# 线程是执行单位
# 启动每一个程序默认都会有一个主线程

# 多线程
from threading import Thread

# # 1.
# t = Thread(target=func)  # func函数
# t.start()

# # 2.
# class MyThread(Thread):
#     def run(self):  # 固定的
#         for i in range(1000):
#             print("子线程：",i)
#
# if __name__ == '__main__':
#     t = MyThread()
#     t.start()
#     for i in range(1000):
#         print("主线程：",i)

# def func(name):
#     for i in range(1000):
#         print(name,i)
#
# if __name__ == '__main__':
#     t1 = Thread(target=func, args=("wzz",))  # args必须位元组
#     t1.start()
#
#     t2 = Thread(target=func, args=("wmx",))  # args必须位元组
#     t2.start()



# 多进程
from multiprocessing import Process

# def func():
#     for i in range(1000):
#         print("子进程：",i)
#
# if __name__ == '__main__':
#     p = Process(target=func)
#     p.start()
#     for i in range(1000):
#         print("主进程：",i)



# # 线程池和进程池
# # 线程池：一次性开辟一些线程，我们用户直接给线程池提交任务，线程任务的调度交给线程池来完成
# from concurrent.futures import ThreadPoolExecutor,ProcessPoolExecutor
#
# def fn(name):
#     for i in range(1000):
#         print(name, i)
#
#
# if __name__ == '__main__':
#     # 创建线程池
#     with ThreadPoolExecutor(50) as t:
#         for i in range(100):
#             t.submit(fn, name=f"线程{i}")
#     # 等待线程池中的任务全部执行完毕，才继续执行（守护）
#     print("over")





# # 线程-北京新发地菜价
# import requests
# import json
# import jsonpath
# from concurrent.futures import ThreadPoolExecutor
#
# print("名称\t最高价\t平均价\t最低价")
# f = open("D:/python爬虫/线程-北京新发地菜价.txt", "w")
#
#
# def main(current):
#     url = "http://www.xinfadi.com.cn/getPriceData.html"
#     headers = {
#         "User-Agent": "Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/86.0.4240.198 Safari/537.36"}
#     data = {"limit": "20",
#             "current": f"{current + 1}",
#             "pubDateStartTime": "",
#             "pubDateEndTime": "",
#             "prodPcatid": "",
#             "prodCatid": "",
#             "prodName": ""}
#     parsePrint(url, headers, data)
#
#
# def parsePrint(url, headers, data):
#     res = requests.post(url, headers=headers, data=data)
#     jsonbj = json.loads(res.text)
#     # 解析数据
#     name = jsonpath.jsonpath(jsonbj, "$..prodName")
#     highPrice = jsonpath.jsonpath(jsonbj, "$..highPrice")
#     avgPrice = jsonpath.jsonpath(jsonbj, "$..avgPrice")
#     lowPrice = jsonpath.jsonpath(jsonbj, "$..lowPrice")
#     # 打印信息
#     for g in range(0, 20):
#         # 输出数据
#         print(f"{name[g]}\t{highPrice[g]}\t{avgPrice[g]}\t{lowPrice[g]}\n")
#         # 写入数据
#         f.write(f"{name[g]}\t{highPrice[g]}\t{avgPrice[g]}\t{lowPrice[g]}\n")
#
#
# if __name__ == '__main__':
#     # 创立线程池
#     with ThreadPoolExecutor(2) as t:
#         for i in range(2):  # 爬取页数
#             # 给子线程提交任务并执行
#             t.submit(main, current=i)
#     print("下载完成")
#     f.close()





# 协程 ：当程序遇见了IO操作，可以选择性的切换到其他任务上
time.sleep(3)
input()
requests.get('bilibili')
# 一般情况，当程序处于IO状态，线程都会处于阻塞状态
# 微光上一个任务一个任务切换
# 宏观上，多个任务一起执行
# 在单线程的条件下