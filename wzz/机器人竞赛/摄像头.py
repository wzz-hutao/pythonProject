# import cv2
# import numpy as np
#
#
# # 调用usb摄像头
# camera_id = 0
# cap = cv2.VideoCapture(camera_id)
#
# # 显示
# while True:
#     ret, frame = cap.read()
#
#
#     frame = cv2.flip(frame,1)
#     cv2.imshow("window", frame)
#     # 如果输入 esc 退出程序
#     boardkey = cv2.waitKey(10) & 0xFF
#     if boardkey == 27:
#         break
# # 关闭
# cap.release()
# cv2.destroyAllWindows()


# import cv2
# vc = cv2.VideoCapture(0)  # 参数为0，则打开摄像头
# if vc.isOpened():
#     open, frame = vc.read()  # 读取第一帧
# else:
#     open = False
#
# while open:
#     ret, frame = vc.read()  # 读取第一帧
#     if frame is None:
#         break
#     if ret:
#         # 将这一帧转换为灰度图
#         # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         frame = cv2.flip(frame, 1)  # 调整是否镜像（1为镜像）
#         cv2.imshow('result', frame)
#         if cv2.waitKey(1) & 0xFF == 27:  # 数字越大越慢（每一帧之间的间隔时长）,按esc退出
#             break
# vc.release()
# cv2.destroyAllWindows()

import cv2
# 人工智能工具包
import mediapipe as mp
# 导入python绘图matplotlib
import matplotlib.pyplot as plt
import numpy as np

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
# img = cv2.imread("D:/robot/OIP-C.jpg")
#
# img_RGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
# results = pose.process(img_RGB)
#
# # 交互式3维可视化
# coords = np.array(results.pose_landmarks.landmark)
#
# def get_y(each):
#     return each.y
#
# points_y = np.array(list(map(get_y,coords)))
#
# print(points_y)



# B-摄像头实时检测
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

# 处理单帧函数
def process_frame(img):
# BGR转RGB
    img_RGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
# 将RGB图像输入模型，获取预测结果
    results = pose.process(img_RGB)

    results_real = results.pose_world_landmarks.landmark[mp_pose.PoseLandmark.NOSE]
    print(results_real)

# 可视化
    mp_drawing.draw_landmarks(img,results.pose_landmarks,mp_pose.POSE_CONNECTIONS)

    return img


# 调用摄像头获取每帧

# 获取摄像头，传入0表示获取系统默认摄像头
cap = cv2.VideoCapture(0)
# 打开cap
cap.open(0)

# 无线循环，知道break被触发
while cap.isOpened():
# 获取画面
    success,frame = cap.read()
    if not success:
        print("ERROR")
        break
# 镜像输出视频
    frame = cv2.flip(frame, 1)
# 处理帧函数
    frame = process_frame(frame)
# 展示处理后的三通道图像
    cv2.imshow('my_window',frame)
    # 按键盘上的q或esc退出(英文输入法下) waitKey(a) a越大越慢(每一帧之间的间隔时长)
    if cv2.waitKey(1) in [ord('q'),27]:
        break
# 关闭摄像头
cap.release()
# 关闭图像窗口
cv2.destroyAllWindows()