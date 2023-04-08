# %matplotlib inline
# opencv-python
import cv2
# 人工智能工具包
import mediapipe as mp
# 导入python绘图matplotlib
import matplotlib.pyplot as plt
import numpy as np
# 进度条库
from tqdm import tqdm
# 时间库
import time

# A-单张图像检测
# 定义可视化图像函数
def look_img(img):
    img_RGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    plt.imshow(img_RGB)
    plt.show()

# 导入solution
mp_pose = mp.solutions.pose

# 导入绘图函数
mp_drawing = mp.solutions.drawing_utils
# 导入模型
pose = mp_pose.Pose(static_image_mode=True, # 是静态图片还是视频
                    model_complexity=2,  # 选择人体姿态关键点检测模型，0性能差但快，2性能好但慢，1介于两者之间
                    smooth_landmarks=True, # 是否平滑关键点
                    enable_segmentation=True, # 是否人体抠图
                    min_tracking_confidence=0.5, # 置信度阈值
                    min_detection_confidence=0.5) # 追踪阈值

# 从图片文件读入图像，opencv读入为BGR模式
img = cv2.imread("D:/robot/R-C.jpg")

# BGR转RGB
img_RGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
# 将RGB图像输入模型，获取预测结果
results = pose.process(img_RGB)
print(results)

mp_drawing.draw_landmarks(img,results.pose_landmarks,mp_pose.POSE_CONNECTIONS)
look_img(img)

# 在3维真实物理坐标系中可视化以米为单位的检测结果
mp_drawing.plot_landmarks(results.pose_world_landmarks,mp_pose.POSE_CONNECTIONS)


# # B-摄像头实时检测
# # 定义可视化图像函数
# def look_img(img):
#     img_RGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
#     plt.imshow(img_RGB)
#     plt.show()
#
# # 导入solution
# mp_pose = mp.solutions.pose
#
# # 导入绘图函数
# mp_drawing = mp.solutions.drawing_utils
# # 导入模型
# pose = mp_pose.Pose(static_image_mode=True, # 是静态图片还是视频
#                     model_complexity=2,  # 选择人体姿态关键点检测模型，0性能差但快，2性能好但慢，1介于两者之间
#                     smooth_landmarks=True, # 是否平滑关键点
#                     enable_segmentation=True, # 是否人体抠图
#                     min_tracking_confidence=0.5, # 置信度阈值
#                     min_detection_confidence=0.5) # 追踪阈值
#
# # 处理单帧函数
# def process_frame(img):
# # BGR转RGB
#     img_RGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
# # 将RGB图像输入模型，获取预测结果
#     results = pose.process(img_RGB)
# # 可视化
#     mp_drawing.draw_landmarks(img,results.pose_landmarks,mp_pose.POSE_CONNECTIONS)
#
#     return img
#
#
# # 调用摄像头获取每帧
#
# # 获取摄像头，传入0表示获取系统默认摄像头
# cap = cv2.VideoCapture(0)
# # 打开cap
# cap.open(0)
#
# # 无线循环，知道break被触发
# while cap.isOpened():
# # 获取画面
#     success,frame = cap.read()
#     if not success:
#         print("ERROR")
#         break
# # 镜像输出视频
#     frame = cv2.flip(frame, 1)
# # 处理帧函数
#     frame = process_frame(frame)
# # 展示处理后的三通道图像
#     cv2.imshow('my_window',frame)
#     # 按键盘上的q或esc退出(英文输入法下) waitKey(a) a越大越慢(每一帧之间的间隔时长)
#     if cv2.waitKey(1) in [ord('q'),27]:
#         break
# # 关闭摄像头
# cap.release()
# # 关闭图像窗口
# cv2.destroyAllWindows()



# # C-单张图像检测+人体抠图+坐标分析+三位交互可视化
# # 定义可视化图像函数
# def look_img(img):
#     img_RGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
#     plt.imshow(img_RGB)
#     plt.show()
#
# # 导入solution
# mp_pose = mp.solutions.pose
#
# # 导入绘图函数
# mp_drawing = mp.solutions.drawing_utils
# # 导入模型
# pose = mp_pose.Pose(static_image_mode=True, # 是静态图片还是视频
#                     model_complexity=2,  # 选择人体姿态关键点检测模型，0性能差但快，2性能好但慢，1介于两者之间
#                     smooth_landmarks=True, # 是否平滑关键点
#                     enable_segmentation=True, # 是否人体抠图
#                     min_tracking_confidence=0.5, # 置信度阈值
#                     min_detection_confidence=0.5) # 追踪阈值
#
#
# img = cv2.imread("D:/robot/R-C.jpg")
#
# img_RGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
# results = pose.process(img_RGB)
#
# mask = results.segmentation_mask
# mask = mask > 0.5
# plt.imshow(mask)
# plt.show()  # 人体区域图
#
# # 单通道转三通道
# mask_3 = np.stack((mask,mask,mask),axis=-1)
# MASK_COLOR = [0,200,0]
# fg_image = np.zeros(img.shape,dtype=np.uint8)
# fg_image[:] = MASK_COLOR
#
# # 获得前景人像
# FG_img = np.where(mask_3,img,fg_image)
#
# # 获得抠掉前景人像的背景
# BG_img = np.where(~mask_3,img,fg_image)
#
# look_img(FG_img)
# look_img(BG_img)
#
# # 所有关键点检测结果
# print(results.pose_landmarks)
# print(mp_pose.POSE_CONNECTIONS) # 33个关键点如何连接
#
# # 左胳膊肘关键点的归一化操作
# results_left1 = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW]
# results_left2 = results.pose_landmarks.landmark[13]
# results_left1_x = results_left1.x
#
# # 原图的像素坐标
# h = img.shape[0]
# w = img.shape[1]
# results_x = results_left1.x * w  # 横
# results_y = results_left1.y * h  # 纵
#
# # 解析指定关键点的真实物理(米)坐标
# results_real = results.pose_world_landmarks.landmark[mp_pose.PoseLandmark.NOSE]  # 鼻子
#
# # 交互式3维可视化
# coords = np.array(results.pose_landmarks.landmark)
# print(len(coords)) # 33个
#
# def get_x(each):
#     return each.x
# def get_y(each):
#     return each.y
# def get_z(each):
#     return each.z
#
# points_x = np.array(list(map(get_x,coords)))
# points_y = np.array(list(map(get_y,coords)))
# points_z = np.array(list(map(get_z,coords)))
#
# # 关键点的坐标
# points = np.vstack([points_x,points_y,points_z]).T
# print(points.shape)
# print(points)
#
# import open3d as o3d
# point_cloud = o3d.geometry.PointCloud()
# point_cloud.points = o3d.utility.Vector3dVector(points)
# o3d.visualization.draw_geometries([point_cloud])



# # D-单张图像检测(优化可视化效果)
# # 定义可视化图像函数
# def look_img(img):
#     img_RGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
#     plt.imshow(img_RGB)
#     plt.show()
#
# # 导入solution
# mp_pose = mp.solutions.pose
#
# # 导入绘图函数
# mp_drawing = mp.solutions.drawing_utils
# # 导入模型
# pose = mp_pose.Pose(static_image_mode=True,  # 是静态图片还是视频
#                     model_complexity=2,   # 选择人体姿态关键点检测模型，0性能差但快，2性能好但慢，1介于两者之间
#                     smooth_landmarks=True,  # 是否平滑关键点
#                     enable_segmentation=True,  # 是否人体抠图
#                     min_tracking_confidence=0.5,  # 置信度阈值
#                     min_detection_confidence=0.5)  # 追踪阈值
#
# # 从图片文件读入图像，opencv读入为BGR模式
# img = cv2.imread("D:/robot/OIP-C.jpg")
#
# # BGR转RGB
# img_RGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
# # 将RGB图像输入模型，获取预测结果
# results = pose.process(img_RGB)
# # 获取左膝盖关键点像素坐标
# h = img.shape[0]
# w = img.shape[1]
#
# cx = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE].x * w)
# cy = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE].y * h)
# cz = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE].z
# print(cx,cy,cz)
#
# # 绘制图：图像，圆心坐标，半径，BGR颜色，最后一个参数为线宽，-1表示填充
# img = cv2.circle(img,(cx,cy),15,(255,0,0),-1)
# look_img(img)
#
# if results.pose_landmarks:  # 若检测出人体关键点
#     # 可视化关键点与骨架连接
#     mp_drawing.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
#
#     for i in range(33):  # 遍历33个关键点
#         # 获取该关键点的三维坐标
#         cx = int(results.pose_landmarks.landmark[i].x * w)
#         cy = int(results.pose_landmarks.landmark[i].y * h)
#         cz = int(results.pose_landmarks.landmark[i].z)
#
#         radius = 10
#
#         if i == 0:  # 鼻尖
#             img = cv2.circle(img, (cx, cy), radius, (0, 0, 255), -1)
#         elif i in [11, 12]:  # 肩膀
#             img = cv2.circle(img, (cx, cy), radius, (223, 155, 6), -1)
#         elif i in [23, 24]:  # 髋关节
#             img = cv2.circle(img, (cx, cy), radius, (1, 240, 255), -1)
#         elif i in [13, 14]:  # 胳膊肘
#             img = cv2.circle(img, (cx, cy), radius, (140, 47, 240), -1)
#         elif i in [25, 26]:  # 膝盖
#             img = cv2.circle(img, (cx, cy), radius, (0, 0, 255), -1)
#         elif i in [15, 16, 27, 28]:  # 手腕和脚腕
#             img = cv2.circle(img, (cx, cy), radius, (223, 155, 60), -1)
#         elif i in [17, 19, 21]:  # 左手
#             img = cv2.circle(img, (cx, cy), radius, (94, 218, 121), -1)
#         elif i in [18, 20, 22]:  # 右手
#             img = cv2.circle(img, (cx, cy), radius, (16, 144, 247), -1)
#         elif i in [27, 29, 31]:  # 左脚
#             img = cv2.circle(img, (cx, cy), radius, (29, 123, 243), -1)
#         elif i in [28, 30, 32]:  # 右脚
#             img = cv2.circle(img, (cx, cy), radius, (193, 182, 255), -1)
#         elif i in [9, 10]:  # 嘴
#             img = cv2.circle(img, (cx, cy), radius, (205, 235, 255), -1)
#         elif i in [1, 2, 3, 4, 5, 6, 7, 8]:  # 脸和脸颊
#             img = cv2.circle(img, (cx, cy), radius, (94, 218, 121), -1)
#         else:  # 其他关键点
#             img = cv2.circle(img, (cx, cy), radius, (0, 255, 0), -1)
#
#     look_img(img)
# else:
#     print("从图像中未检测出人体关键点，报错")
#
# # 保存图片
# cv2.imwrite('D:/robot/OIP-C1.jpg',img)  # 位置 图像


# # E-摄像头实时检测(高阶)
# # 定义可视化图像函数
# def look_img(img):
#     img_RGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
#     plt.imshow(img_RGB)
#     plt.show()
#
# # 导入solution
# mp_pose = mp.solutions.pose
#
# # 导入绘图函数
# mp_drawing = mp.solutions.drawing_utils
# # 导入模型
# pose = mp_pose.Pose(static_image_mode=True,  # 是静态图片还是视频
#                     model_complexity=0,   # 选择人体姿态关键点检测模型，0性能差但快，2性能好但慢，1介于两者之间
#                     smooth_landmarks=True,  # 是否平滑关键点
#                     enable_segmentation=True,  # 是否人体抠图
#                     min_tracking_confidence=0.5,  # 置信度阈值
#                     min_detection_confidence=0.5)  # 追踪阈值
#
# def process_frame(img):
#     # 记录该帧开始处理的时间
#     start_time = time.time()
#
#     # 获取图像宽高
#     h, w = img.shape[0],img.shape[1]
#     # BGR转RGB
#     img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     # 将RGB图像输入模型，获得预测结果
#     results = pose.process(img_RGB)
#
#     if results.pose_landmarks:  # 若检测出人体关键点
#         # 可视化关键点与骨架连接
#         mp_drawing.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
#
#         for i in range(33):  # 遍历33个关键点
#             # 获取该关键点的三维坐标
#             cx = int(results.pose_landmarks.landmark[i].x * w)
#             cy = int(results.pose_landmarks.landmark[i].y * h)
#             cz = int(results.pose_landmarks.landmark[i].z)
#
#             radius = 10
#
#             if i == 0:  # 鼻尖
#                 img = cv2.circle(img, (cx, cy), radius, (0, 0, 255), -1)
#             elif i in [11, 12]:  # 肩膀
#                 img = cv2.circle(img, (cx, cy), radius, (223, 155, 6), -1)
#             elif i in [23, 24]:  # 髋关节
#                 img = cv2.circle(img, (cx, cy), radius, (1, 240, 255), -1)
#             elif i in [13, 14]:  # 胳膊肘
#                 img = cv2.circle(img, (cx, cy), radius, (140, 47, 240), -1)
#             elif i in [25, 26]:  # 膝盖
#                 img = cv2.circle(img, (cx, cy), radius, (0, 0, 255), -1)
#             elif i in [15, 16, 27, 28]:  # 手腕和脚腕
#                 img = cv2.circle(img, (cx, cy), radius, (223, 155, 60), -1)
#             elif i in [17, 19, 21]:  # 左手
#                 img = cv2.circle(img, (cx, cy), radius, (94, 218, 121), -1)
#             elif i in [18, 20, 22]:  # 右手
#                 img = cv2.circle(img, (cx, cy), radius, (16, 144, 247), -1)
#             elif i in [27, 29, 31]:  # 左脚
#                 img = cv2.circle(img, (cx, cy), radius, (29, 123, 243), -1)
#             elif i in [28, 30, 32]:  # 右脚
#                 img = cv2.circle(img, (cx, cy), radius, (193, 182, 255), -1)
#             elif i in [9, 10]:  # 嘴
#                 img = cv2.circle(img, (cx, cy), radius, (205, 235, 255), -1)
#             elif i in [1, 2, 3, 4, 5, 6, 7, 8]:  # 脸和脸颊
#                 img = cv2.circle(img, (cx, cy), radius, (94, 218, 121), -1)
#             else:  # 其他关键点
#                 img = cv2.circle(img, (cx, cy), radius, (0, 255, 0), -1)
#
#         # look_img(img)
#     else:
#         scaler = 1
#         failure_str = 'No Person'
#         img = cv2.putText(img, failure_str, (25 * scaler,100 * scaler), cv2.FONT_HERSHEY_SIMPLEX,
#                           1.25 * scaler,(255, 0, 0))
#     end_time = time.time()
#     FPS = 1/(end_time - start_time)
#
#     scaler = 1
#     mg = cv2.putText(img, 'FPS   '+str(int(FPS)), (25 * scaler, 50 * scaler), cv2.FONT_HERSHEY_SIMPLEX,
#                      1.25 * scaler, (255, 0, 0))
#     return img
#
#
# # 调用摄像头获取每帧
#
# # 获取摄像头，传入0表示获取系统默认摄像头
# cap = cv2.VideoCapture(0)
# # 打开cap
# cap.open(0)
#
#
# # 无限循环，直到break被触发
# while cap.isOpened():
# # 获取画面
#     success,frame = cap.read()
#     if not success:
#         print("ERROR")
#         break
# # 镜像输出视频
#     frame = cv2.flip(frame, 1)
# # 处理帧函数
#     frame = process_frame(frame)
# # 展示处理后的三通道图像
#     cv2.imshow('my_window',frame)
#     if cv2.waitKey(1) in [ord('q'),27]: # 按键盘上的q或esc退出(英文输入法下)
#         break
# # 关闭摄像头
# cap.release()
# # 关闭图像窗口
# cv2.destroyAllWindows()



# # F-视频检测
# # 定义可视化图像函数
# def look_img(img):
#     img_RGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
#     plt.imshow(img_RGB)
#     plt.show()
#
# # 导入solution
# mp_pose = mp.solutions.pose
#
# # 导入绘图函数
# mp_drawing = mp.solutions.drawing_utils
# # 导入模型
# pose = mp_pose.Pose(static_image_mode=False,  # 是静态图片还是视频
#                     model_complexity=2,   # 选择人体姿态关键点检测模型，0性能差但快，2性能好但慢，1介于两者之间
#                     smooth_landmarks=True,  # 是否平滑关键点
#                     enable_segmentation=True,  # 是否人体抠图
#                     min_tracking_confidence=0.5,  # 置信度阈值
#                     min_detection_confidence=0.5)  # 追踪阈值
#
#
# def process_frame(img):
#     # 记录该帧开始处理的时间
#     start_time = time.time()
#
#     # 获取图像宽高
#     h, w = img.shape[0],img.shape[1]
#     # BGR转RGB
#     img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     # 将RGB图像输入模型，获得预测结果
#     results = pose.process(img_RGB)
#
#     if results.pose_landmarks:  # 若检测出人体关键点
#         # 可视化关键点与骨架连接
#         mp_drawing.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
#
#         for i in range(33):  # 遍历33个关键点
#             # 获取该关键点的三维坐标
#             cx = int(results.pose_landmarks.landmark[i].x * w)
#             cy = int(results.pose_landmarks.landmark[i].y * h)
#             cz = int(results.pose_landmarks.landmark[i].z)
#
#             radius = 10
#
#             if i == 0:  # 鼻尖
#                 img = cv2.circle(img, (cx, cy), radius, (0, 0, 255), -1)
#             elif i in [11, 12]:  # 肩膀
#                 img = cv2.circle(img, (cx, cy), radius, (223, 155, 6), -1)
#             elif i in [23, 24]:  # 髋关节
#                 img = cv2.circle(img, (cx, cy), radius, (1, 240, 255), -1)
#             elif i in [13, 14]:  # 胳膊肘
#                 img = cv2.circle(img, (cx, cy), radius, (140, 47, 240), -1)
#             elif i in [25, 26]:  # 膝盖
#                 img = cv2.circle(img, (cx, cy), radius, (0, 0, 255), -1)
#             elif i in [15, 16, 27, 28]:  # 手腕和脚腕
#                 img = cv2.circle(img, (cx, cy), radius, (223, 155, 60), -1)
#             elif i in [17, 19, 21]:  # 左手
#                 img = cv2.circle(img, (cx, cy), radius, (94, 218, 121), -1)
#             elif i in [18, 20, 22]:  # 右手
#                 img = cv2.circle(img, (cx, cy), radius, (16, 144, 247), -1)
#             elif i in [27, 29, 31]:  # 左脚
#                 img = cv2.circle(img, (cx, cy), radius, (29, 123, 243), -1)
#             elif i in [28, 30, 32]:  # 右脚
#                 img = cv2.circle(img, (cx, cy), radius, (193, 182, 255), -1)
#             elif i in [9, 10]:  # 嘴
#                 img = cv2.circle(img, (cx, cy), radius, (205, 235, 255), -1)
#             elif i in [1, 2, 3, 4, 5, 6, 7, 8]:  # 脸和脸颊
#                 img = cv2.circle(img, (cx, cy), radius, (94, 218, 121), -1)
#             else:  # 其他关键点
#                 img = cv2.circle(img, (cx, cy), radius, (0, 255, 0), -1)
#
#         # look_img(img)
#     else:
#         scaler = 1
#         failure_str = 'No Person'
#         img = cv2.putText(img, failure_str, (25 * scaler,100 * scaler), cv2.FONT_HERSHEY_SIMPLEX,
#                           1.25 * scaler,(255, 0, 0))
#     end_time = time.time()
#     FPS = 1/(end_time - start_time)
#
#     scaler = 1
#     mg = cv2.putText(img, 'FPS   '+str(int(FPS)), (25 * scaler, 50 * scaler), cv2.FONT_HERSHEY_SIMPLEX,
#                      1.25 * scaler, (255, 0, 0))
#     return img
#
#
# def generate_video(input_path=''):
#     filehead = input_path.split('/')[-1]
#     output_path = 'D:/robot/out' + filehead
#
#     print("视频开始处理",input_path)
#     # 获取视频总帧数
#     cap = cv2.VideoCapture(input_path)
#     frame_count = 0
#     while(cap.isOpened()):
#         success, frame = cap.read()
#         frame_count += 1
#         if not success:
#             break
#     cap.release()
#     print("视频总帧数为",frame_count)
#
#     cap = cv2.VideoCapture(input_path)
#     frame_size = (cap.get(cv2.CAP_PROP_FRAME_WIDTH),cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#     fps = cap.get(cv2.CAP_PROP_FPS)
#
#     out = cv2.VideoWriter(output_path,fourcc,fps,(int(frame_size[0]),int(frame_size[1])))
#
#     # 进度条绑定视频总帧数
#     with tqdm(total=frame_count-1) as pbar:
#         try:
#             while(cap.isOpened()):
#                 success, frame = cap.read()
#                 frame_count += 1
#                 if not success:
#                     break
#                 # 处理帧
#                 try:
#                     frame = process_frame(frame)
#                 except:
#                     print('error')
#                     pass
#
#                 if success == True:
#                     out.write(frame)
#                     # 进度条更新一帧
#                     pbar.update(1)
#
#         except:
#             print("中途中断")
#             pass
#
#     cv2.destroyAllWindows()
#     out.release()
#     cap.release()
#     print("视频已保存",output_path)
#
# generate_video(input_path="D:/robot/sdsp.mp4")



