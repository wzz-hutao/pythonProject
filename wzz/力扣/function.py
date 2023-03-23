# # zip()
# strs = ["flower","flow","flight"]
# for tmp in zip(*strs):   # tmp = ('f','f','f'),('l','l','l'),('o','o','i')...


# # enumerate()
# names = ["Alice","Bob","Carl"]
# for index,value in enumerate(names):
#     print(f'{index}: {value}')
# 0: Alice
# 1: Bob
# 2: Carl


# find()
# str.find(str, beg=0, end=len(string):
# str : 指定检索的字符串
# beg : 开始索引，默认为0。
# end : 结束索引，默认为字符串的长度。
# 找到了返回起始下标，找不到返回-1


# stack
# st = list()
# st.append() = push_back # 放在最后
# st.pop() # 删除最后一个元素，即栈顶元素
# st[-1] = top() # 相当于栈顶


# 字符串大小写转化
# strs = strs.lower()
# strs = strs.upper()


# 以字符串的形式连续输出列表内容
# list1 = ['a','b','c']
# print("".join(list1[::-1]))  # cba


# 数字转字符 “A” - 1
# chr(a0 - 1 + ord("A") # 将a0变为字符


# /,//,%
# 5 / 2 = 2.5
# 5 // 2 = 2(除法只取整数部分)
# 5 % 2 = 1 # 取余数


# ret, start = [], 0
# ret.append(f"{nums[start]}->{nums[i - 1]}") # “1->3"


# for i in range(len(columnTitle) - 1, -1, -1):   倒序


# ord 将字符转换为ASCII value
# Input:11
# val = 'A'
# print(ord(val))
# Output:  65


# nums1 = [1,2,2,1]
# m = collections.Counter()
# for num in nums1:
#     m[num] += 1
# # m : Counter({1:2,2:2}) 计数器


# # if (count := m.get(num, 0)) > 0:
# a = 7
# print(v := a+2)  # 9   v = 9 直接赋值
# m.get(num,s)  # 字典的get方法,在m中寻找num的键，如果在，返回键值，如果不在，返回s


# import collections
# s = "abscfff"
# t = "abscgftrht"
# print(collections.Counter(s))   # Counter({'f': 3, 'a': 1, 'b': 1, 's': 1, 'c': 1})
# print(collections.Counter(s) - collections.Counter(t))  # Counter({'f': 2})


# s = ['h','e','l','l','o']
# "".join(s) -> hello
# " ".join(s) -> h e l l o


# # 二维数组遍历
# num = [[5,10],[6,8],[1,5],[2,3],[1,10]]
# for i,j in num:
#     print(f"{[i,j]}")


# import collections
#
# num = [1,2,3,3,3,4,4]
# cnt = collections.Counter(num)
# for key,val in cnt.items():
#     print(f"[{key},{val}]",end=" ")
# #  [1,1] [2,1] [3,3] [4,2]


# 堆
# heapq


# 定义数组
# num = [0] * 10
# g = [[] for _ in range(n)]


# 队列
# q = collections.deque()


# dp = [[0 for _ in range(n)] for _ in range(m)]
# [[0,0,0],[0,0,0],[0,0,0]]


# 返回一个二维数组(按照a排序)
# map = collections.Counter()
# sorted([a, b] for a, b in map.items())


# # 直接插入排序
# def insertion_sort(array):
#     for i in range(len(array)):
#         cur_index = i
#         while array[cur_index - 1] > array[cur_index] and cur_index - 1 >= 0:
#             array[cur_index], array[cur_index - 1] = array[cur_index - 1], array[cur_index]
#             cur_index -= 1
#     return array
#
#
# if __name__ == '__main__':
#     array = [10, 17, 50, 7, 30, 24, 27, 45, 15, 5, 36, 21]
#     print(insertion_sort(array))


# # 希尔排序
# def shell_sort(array, n):
#     interval = int(len(array) / n)
#     while interval > 0:
#         for i in range(interval, len(array)):
#             cur_index = i - interval
#             while cur_index >= 0 and array[cur_index] > array[cur_index + interval]:
#                 array[cur_index + interval], array[cur_index] = array[cur_index], array[cur_index + interval]
#                 cur_index -= interval
#         interval = int(interval / n)
#     return array
#
#
# if __name__ == '__main__':
#     array = [10, 17, 50, 7, 30, 24, 27, 45, 15, 5, 36, 21]
#     print(shell_sort(array,3))


# # 冒泡排序
# def Bubblesort(array):
#     n = len(array)
#     exchange = False
#     for i in range(n):
#         for j in range(n-1,i+1,-1):
#             exchange = False
#             if array[j] < array[j-1]:
#                 array[j], array[j-1] = array[j-1], array[j]
#                 exchange = True
#         if exchange == False:
#             break
#     return array
#
# if __name__ == '__main__':
#     array = [1,3,2,6,4,7,9,8]
#     print(Bubblesort(array))


# # 26个字母 大小写
# char_dx = [chr(i) for i in range(65, 91)]
# char_xx = [chr(i) for i in range(97, 123)]

# import bisect
# import itertools
# class Solution:
#     def answerQueries(self, nums: List[int], queries: List[int]) -> List[int]:
#         f = list(itertools.accumulate(sorted(nums)))  # [1,3,7,12]
#         return [bisect.bisect_right(f, q) for q in queries]
#
# a = Solution()
# nums = [4,5,2,1]
# queries = [3,10,21]
# print(a.answerQueries(nums,queries))

# from typing import List


# divmod(a,b)
# (a // b,a % b)


# grid = [[0,1],[2,3]]
# pos = [None] * (len(grid) ** 2)
# for i, row in enumerate(grid):
#     for j, x in enumerate(row):
#         pos[x] = (i,j)
# print(pos) # [(0, 0), (0, 1), (1, 0), (1, 1)] 位置


# pairwase() 值相邻的


# # # 回溯
# class Solution:
#     def subsets(self, nums: list[int]) -> list[list[int]]:
#         ans = []
#         path = []
#         n = len(nums)
#         def dfs(i):
#             if i == n:
#                 ans.append(path.copy())
#                 return
#             dfs(i+1)
#
#             path.append(nums[i])
#             dfs(i+1)
#             path.pop()
#         dfs(0)
#         return ans


# # 数字字符串转换为列表
# N = 123
# s = list(map(int, str(N)))
# print(s)  #  [1, 2, 3]


# import math
# class Solution:
#     def numDupDigitsAtMostN(self, n: int) -> int:
#         m, s = list(map(int,str(n+1))), set()
#         res = sum(9 * math.perm(9,i) for i in range(len(m) - 1))
#         for i, x in enumerate(m):
#             for j in range(i == 0,x):
#                 if j not in s:
#                     res += math.perm(9-i,len(m)-i-1)
#             if x in s:
#                 break
#             s.add(x)
#         return n - res
#
# a = Solution()
# print(a.numDupDigitsAtMostN(110))


# scores = [4,5,6,5]
# ages = [2,1,2,1]
# people = sorted(zip(scores, ages))
# print(people)  # [(4, 2), (5, 1), (5, 1), (6, 2)] 按照scores[i]排序



