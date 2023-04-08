def print_file_info(file_name):
    """
    功能：将给定路径的文件内容输出到控制台
    :param file_name: 读取的文件路径
    :return:
    """
    f = None
    try:
        f = open(file_name, "r",encoding="UTF-8")
    except Exception as e:
        print(f"出现异常,异常是{e}")
    else:
        print(f.read())
    finally:
        if f:
            f.close()

def append_to_file(file_name,data):
    """
    功能：将指定的数据追加到指定的文件中
    :param file_name: 指定的文件路径
    :param data: 指定的数据
    :return: None
    """
    f = open(file_name,"a",encoding="UTF-8")
    f.write("\n")
    f.write(data)
    f.close()


if __name__ == '__main__':
    # print_file_info("D:/新建文件夹 (2)/test_py.txt")
    append_to_file("D:/新建文件夹 (2)/test_append.txt","ikun")