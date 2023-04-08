# # SQL
# from pymysql import Connection
# conn = Connection(
#     host = 'LAPTOP-N1VJVV1I',
#     port = 3306,
#     user= 'LAPTOP-N1VJVV1I\wzzyyds',
#     password = '123456'
# )
# print(conn.get_server_info())
# conn.close()


# PySpark
# import pyspark


# # 网络编程
# # 服务端开发
# import socket
# # 创建socket对象
# socket_server = socket.socket()
# # 绑定ip地址和端口
# socket_server.bind(("localhost",8888))
# # 监听窗口
# socket_server.listen(1)
# # listen(a)   a为整数，表示接收的链接数量
# # 等待客户端连接
# conn,address = socket_server.accept()
# # accept方法返回的是二元元组(链接对象，客户端地址信息)
# # accept方法如果没有客户端连接，会直接卡住不向下执行
# print(f"接收到了客户端，客户端的信息是：{address}")
# # 接收客户端信息
# conn.recv(1024).decode("UTF-8")
# # recv()缓冲区大小，一般为1024，返回值为一个字节数组，可以通过decode()方法通过UTF-8编码，将字节数组转换为字符串对象
# # 发送回复消息
# msg = input("请输入你要和客户端回复的消息:").encode("UTF-8")  # 将字符串编码转换为字节数组对象
# conn.send(msg)
# # 关闭链接
# conn.close()
# socket_server.close()


