#pta
# 2-5-1
# def fact(n):
#     if n <= 1:
#         return 1
#     else:
#         return n*fact(n-1)
#
# a = int(input())
# b = int(input())
# result = 0
# if a>=0 & b>=0:
#     result = fact(b)/fact(b-a)
# print(result)



# 2-5-5
# def mypow1(n,m):
#     if m == 0:
#         return 1
#     if m == 1:
#         return n
#     count = mypow(n,m/2)
#     if m%2 == 1:
#         return count*count*n
#     else:
#         return count*count

# def mypow2(n,m):
#     if m == 0:
#         return 1
#     if m == 1:
#         return n
#     count = mypow(n,m-1)
#     return count*n

# a = 0.24
# b = 4
# c = mypow2(a,b)
# print(c)



# 5-1
# def dist(x1,x2,y1,y2):
#     return ((x1-x2)**2 + (y1-y2)**2)**0.5
#
# x1,x2,y1,y2 = 3,4,5,8
#
# print(dist(x1, x2, y1, y2))

# 5-3
# def fib(n):
#     if n == 1:
#         return 1
#     elif n == 2:
#         return 1
#     else:
#         return fib(n-1)+fib(n-2)
#
# a = int(input())
# print(fib(a))

#5-5
# def gcd(n,m):
#     c = 1
#     while c:
#         c = n%m
#         n = m
#         m = c
#         c = n%m
#     return m
#
# a,b = map(int,input().split())
# print(gcd(a,b))

# 5-6
# import math
# def prime(n):
#     sign = True
#     if n <= 1: return False
#     if n == 2 :return sign
#     else :
#         s = math.sqrt(n)
#         for i in range(2,int(s)+1):
#             if n%i==0 :
#                 sign = False
#     return sign
#
# def prime_sum(a,b):
#     count = 0
#     for i in range(a,b+1):
#         if prime(i) == 1:
#             count+=i
#     return count
#
# a,b = map(int,input().split())
# print(prime_sum(a,b))

# 5-7
# a,b = map(int,input().split())
# c = input()
# for i in range(b):
#     for j in range(a):
#         print(c,end = "")
#     print(" ")

#
# import math
# def funcos(e,x):
#     item = 1
#     i = 2
#     jiecheng = 1
#     count = 1
#     f = -1
#     p = 1
#     while item > e:
#         for j in range(1,i+1):
#             jiecheng *= j
#         p = math.pow(x,i)
#         item = p / jiecheng
#         count += f*item
#         i+=2
#         f = -f
#         jiecheng = 1
#     return count
#
# a,b = map(float,input().split())
# print(funcos(a,b))

# 5-10
# def countdigit(n,m):
#     count = 0
#     while n:
#         if n % 10 == m:
#             count+=1
#         n/=10
#     return count
#
# n,m = map(int,input().split())
# print(countdigit(n,m))

# # 6-3
# def fn(a,n):
#     i = a
#     n-=1
#     while n:
#         i += a*10
#         a *= 10
#         n-=1
#     return i
#
# def SumA(a,n):
#     sum = 0
#     while n:
#         sum += fn(a,n)
#         n-=1
#     return sum
#
# a,b = map(int,input().split())
# print(SumA(a,b))

# 6-7
# import math
# def reverse(num):
#     flag = 0
#     if num < 0:
#         num = -num
#         flag = 1
#     arr = list()
#     i = 0
#     while num:
#         arr.append(num % 10)
#         num = int(num/10)
#         i+=1
#     sum = 0
#     i-=1
#     for j in arr:
#         sum += math.pow(10,i)*j
#         i-=1
#     if not flag : return int(sum)
#     else: return int(-sum)

# a = int(input())
# print(reverse(a))

# 方法2
# a = input('请输入一个整数:')
# b = int(a[::-1])
# print('该数的逆序数为:',b)

# 方法3
# i = int(input('请输入一个整数:'))
# s = str(i)
# l = len(s)
# a = []
# print('该数的逆序数为:',end='')
# for b in range(l):
#     a.append(s[l-1-b])
# for c in a:
#     print(c,end='')

# import numpy as np
# MyType=np.dtype({
#     'names':['name','score1','score2','score3','avg'],
#     'formats':['S32','i','i','i','f'] #必须加s，且S大写
# })
# # c = np.array([('ikun',12,23,34,45.56)],dtype=MyType)
# # print(c['avg'])
# def choose_sort_key(element):
#     return element['avg']
# def main():
#     while True:
#         try:
#             a = int(input())
#             list = []
#             for i in range(a):
#                 name = input()
#                 score1 = int(input())
#                 score2 = int(input())
#                 score3 = int(input())
#                 avg = (score1 + score2 + score3) / 3.0
#                 b = np.array([(name,score1,score2,score3,avg)],dtype=MyType)
#                 list.append(b)
#             list.sort(key = choose_sort_key())
#             print(list)
#
#         except EOFError:
#             break
#
# main()














