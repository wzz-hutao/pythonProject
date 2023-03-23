# import os
#
# def test_os():
#
#     # 列出路径下的内容
#     print(os.listdir("D:/资料"))
#     # 判断指定路径是否为文件夹
#     print(os.path.isdir("D:/资料/物理"))
#     # 判断路径是否存在
#     print(os.path.exists("D:/资料/概率论"))
#
# def get_files_recursion_from_dir(path):
#     """
#     使用递归获取全部的文件列表
#     :param path: 被判断的文件夹
#     :return: list
#     """
#     file_list = []
#     if os.path.exists(path):
#         for f in os.listdir():
#             new_path = path + "/" + f
#             if os.path.isdir(new_path):
#                 file_list += get_files_recursion_from_dir(new_path)
#             else:
#                 file_list.append(new_path)
#
#     else:
#         print("文件路径不存在")
#         return []
#     return file_list
#
#
# if __name__ == '__main__':
#     print(get_files_recursion_from_dir("D:/资料"))