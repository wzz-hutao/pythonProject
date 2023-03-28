# class Solution:
#     def longestCommonPrefix(self, s: list[str]) -> str:
#         if not s:
#             return ""
#         res = s[0]
#         i = 1
#         while i < len(s):
#             while s[i].find(res) != 0:
#                 res = res[0:len(res)-1]
#             i += 1
#         return res
#
#
#
#
# a = Solution()
# strs = ["flow","flower","fl"]
# print(a.longestCommonPrefix(strs))


# class Solution(object):
#     def isValid(self, s):
#         """
#         :type s: str
#         :rtype: bool
#         """
#         n = len(s)
#         if n % 2 == 1:
#             return False
#         q = {")":"(",
#              "]":"[",
#              "}":"{"}
#         st = list()
#         for strs in s:
#             if strs in q.keys():
#                 if not st or st[-1] != q[strs]:
#                     return False
#                 else:
#                     st.pop()
#             else:
#                 st.append(strs)
#         return not st
#
# a = Solution()
# strs = input()
# print(a.isValid(strs))



# import re
#
#
# class Solution:
#     def isPalindrome(self, s: str) -> bool:
#         t = s.lower()
#         r = r'[a-zA-Z]'
#         result = re.findall(r,t)
#         right = len(result)-1
#         left = 0
#         while left<=right:
#             if result[left] != result[right]:
#                 return False
#             left+=1
#             right-=1
#         return True
#
# a = Solution()
# strs = input()
# print(a.isPalindrome(strs))


# class Solution:
#     def missingNumber(self, nums: list[int]) -> int:
#         nums.sort()
#         for i in range(len(nums)):
#             if nums[i] != i:
#                 return i
#         return len(nums)
#
# a = Solution()
# nums = [0]
# print(a.missingNumber(nums))


# class Solution:
#     def generate(self, numRows):
#         ret = list()
#         for i in range(numRows+1):
#             row = list()
#             for j in range(0, i + 1):
#                 if j == 0 or j == i:
#                     row.append(1)
#                 else:
#                     row.append(ret[i - 1][j] + ret[i - 1][j - 1])
#             ret.append(row)
#         return ret[numRows]
#
#
#
# a = Solution()
# num = int(input())
# print(a.generate(num))



# class Solution:
#     def maxProfit(self, prices):
#         inf = int(1e9)
#         minprice = inf
#         maxprofit = 0
#         for price in prices:
#             maxprofit = max(price - minprice, maxprofit)
#             minprice = min(price, minprice)
#         return maxprofit
#
#
# a = Solution()
# nums = [2,4,1]
# print(a.maxProfit(nums))



# class MyCircularQueue(object):
#
#     def __init__(self,k):
#         self.front = self.rear = 0
#         self.element = [0] * (k+1)
#
#     def enQueue(self,value):
#         if self.isFull():
#             return False
#         self.element[self.rear] = value
#         self.rear = (self.rear+1) % len(self.element)
#         return True
#
#     def deQueue(self):
#         if self.isEmpty():
#             return False
#         self.front = (self.front + 1) % len(self.element)
#         return True
#
#     def Front(self):
#         return -1 if self.isEmpty() else self.element[self.front]
#
#     def Rear(self):
#         return -1 if self.isEmpty() else self.element[self.rear-1]
#
#     def isEmpty(self):
#         return self.front == self.rear
#
#     def isFull(self):
#         return (self.rear+1) % len(self.element) == self.front

# import json
#
# class ListNode(object):
#
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
#
# class Solution(object):
#
#     def deleteDuplicates(self, head):
#         """
#         :type head: ListNode
#         :rtype: ListNode
#         """
#         if not head:
#             return head
#         cur = head
#         while cur.next:
#             if cur.next.val == cur.val:
#                 cur.next = cur.next.next
#             else:
#                 cur = cur.next
#         return head
#
#
# def stringToListNode(input):
#     # Generate list from the input
#     numbers = json.loads(input)  # 转换为列表
#
#     # Now convert that list into linked list
#     dummyRoot = ListNode(0)
#     ptr = dummyRoot
#     for number in numbers:
#         ptr.next = ListNode(number)
#         ptr = ptr.next
#
#     ptr = dummyRoot.next
#     return ptr
#
#
# def listNodeToString(node):
#     if not node:
#         return "[]"
#
#     result = ""
#     while node:
#         result += str(node.val) + ", "
#         node = node.next
#     return "[" + result[:-2] + "]"
#
#
# def main():
#     import sys
#     def readlines():
#         for line in sys.stdin:
#             yield line.strip('\n')
#
#     lines = readlines()
#     while True:
#         try:
#             line = next(lines)  # 使用line = lines.next()报错
#             head = stringToListNode(line)
#
#             ret = Solution().deleteDuplicates(head)
#
#             out = listNodeToString(ret)
#             print(out)
#         except StopIteration:
#             break
#
#
# if __name__ == '__main__':
#     main()


# class Solution:
#     def removeElements(self, head: ListNode, val: int) -> ListNode:
#         if not head: return
#         head.next = self.removeElements(head.next, val)
#         return head.next if head.val == val else head




# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
#
#
#
#     def binaryTreePaths(self, root: Optional[TreeNode]) -> List[str]:
#         def construct_paths(root, path):
#             if root:
#                 path += str(root.val)
#                 if not root.left and not root.right:  # 当前节点是叶子节点
#                     paths.append(path)  # 把路径加入到答案中
#                 else:
#                     path += '->'  # 当前节点不是叶子节点，继续递归遍历
#                     construct_paths(root.left, path)
#                     construct_paths(root.right, path)
#
#         paths = []
#         construct_paths(root, '')
#         return paths


# class Solution:
#     def summaryRanges(self, nums):
#         nums.append(2 ** 32)
#         ret, start = [], 0
#         for i in range(1,len(nums)):
#             if nums[i] - nums[i - 1] > 1:
#                 if i - 1 == start:
#                     ret.append(str(nums[start]))
#                 else:
#                     ret.append(f"{nums[start]}->{nums[i - 1]}")
#                 start = i
#         return ret
#
#
# list1 = [0,1,2,4,5,7]
# a = Solution()
# print(a.summaryRanges(list1))



# class Solution:
#     def moveZeroes(self, nums: list[int]) -> None:
#         """
#         Do not return anything, modify nums in-place instead.
#         """
#         for j in range(1,len(nums)):
#             for i in range(j, len(nums)):
#                 if nums[i - 1] == 0 and nums[i] != 0:
#                     nums[i - 1], nums[i] = nums[i], nums[i - 1]
#         return nums
#
#
# a = Solution()
# nums = [0,1,0,3,12]
# print(a.moveZeroes(nums))

# class NumArray:
#
#     def __init__(self, nums: list[int]):
#         self.sums = [0]
#         _sums = self.sums
#
#         for num in nums:
#             _sums.append(_sums[-1] + num)
#
#     def sumRange(self, i: int, j: int) -> int:
#         _sums = self.sums
#         return _sums[j + 1] - _sums[i]
#
# n = int(input())
# num = [0]*n
# for i in range(n):
#     num[i] = int(input())
# numArray = NumArray(num)
# print(numArray.sumRange(0,2))


# class Solution:
#     def wordPattern(self, pattern: str, s: str) -> bool:
#         word2ch = dict()
#         ch2word = dict()
#         words = s.split()
#         if len(pattern) != len(words):
#             return False
#
#         for ch, word in zip(pattern, words):
#             if (word in word2ch and word2ch[word] != ch) or (ch in ch2word and ch2word[ch] != word):
#                 return False
#             word2ch[word] = ch
#             ch2word[ch] = word
#
#         return True
#
#
#
# a = Solution()
# pattern,s = "abba","dog dog dog dog"
# print(a.wordPattern(pattern,s))

# import collections
#
# class Solution:
#
#     def intersect(self, nums1: list[int], nums2: list[int]) -> list[int]:
#         num1 = collections.Counter(nums1)
#         num2 = collections.Counter(nums2)
#         num = num1 & num2
#         return list(num.elements())
#
#
# a = Solution()
# nums1,nums2 = [1,2,2,1],[2,2]
# print(a.intersect(nums1,nums2))



# class Solution:
#
#     def countBits(self, n: int) -> list[int]:
#
#         def count_bit_one(a):
#             count = 0
#             while a:
#                 if a % 2 == 1:
#                     count += 1
#                 a //= 2
#             return count
#
#         nums = []
#         for i in range(n+1):
#             nums.append(count_bit_one(i))
#         return nums


# class Solution:
#     def isPowerOfTwo(self, n: int) -> bool:
#         return n > 0 and (n & (n - 1)) == 0
#
# class Solution:
#     def isPowerOfTwo(self, n: int) -> bool:
#         return n > 0 and (n & -n) == n
#
#
# a = Solution()
# s = int(input())
# print(a.isPowerOfTwo(s))


# import collections

# class Solution:
#     def findTheDifference(self, s: str, t: str) -> str:
#         return list(collections.Counter(t) - collections.Counter(s))[0]
#
# a = Solution()
# s,t = "abcde","abcd"
# print(a.findTheDifference(t,s))

# import collections
#
# class Solution:
#     def longestPalindrome(self, s: str) -> int:
#         t = dict(collections.Counter(s))
#         count,max1 = 0,0
#         for i in t.keys():
#             if t[i] % 2 == 0:
#                 count += t[i]
#             else:
#                 max1 = max(max1,t[i])
#         return count + max1
#
# a = Solution()
# s = "aaaaaccc"
# print(a.longestPalindrome(s))


# class Solution:
#     def reverseVowels(self, s: str) -> str:
#         left,right = 0,len(s)-1
#         s = list(s)
#         vowel = {"a","e","i","o","u","A","E","I","O","U"}
#         while left < right:
#             if s[left] not in vowel:
#                 left += 1
#             if s[right] not in vowel:
#                 right -= 1
#             if s[left] in vowel and s[right] in vowel:
#                 s[left],s[right] = s[right],s[left]
#                 left += 1
#                 right -= 1
#         return "".join(s)
#
#
# a = Solution()
# s = input()
# print(a.reverseVowels(s))


# class Solution:
#     def isSubsequence(self, s: str, t: str) -> bool:
#         n, m = len(s), len(t)
#         i = j = 0
#         while i < n and j < m:
#             if s[i] == t[j]:
#                 i += 1
#             j += 1
#         return i == n
#
# a = Solution()
# s,t = "ab","baab"
# print(a.isSubsequence(s,t))


# import collections
#
# class Solution:
#     def canConstruct(self, ransomNote: str, magazine: str) -> bool:
#         t = len(magazine) - len(ransomNote)
#         s = dict(collections.Counter(magazine) - collections.Counter(ransomNote))
#         count = 0
#         for i in s.keys():
#             count += s[i]
#         return count == t
#
# a = Solution()
# ran,mag = 'aa','aab'
# print(a.canConstruct(ran,mag))



# class Solution:
#     def addDigits(self, num: int) -> int:
#         def count(n):
#             c = 0
#             while n:
#                 c += n % 10
#                 n //= 10
#             return c
#
#         while num > 9:
#             num = count(num)
#         return num
#
#
# a = Solution()
# s = int(input())
# print(a.addDigits(s))


# class Solution:
#     def findContentChildren(self, g: list[int], s: list[int]) -> int:
#         g.sort()
#         s.sort()
#         i, j = 0, 0
#         while i < len(g) and j < len(s):
#             if s[j] >= g[i]:
#                 j += 1
#                 i += 1
#             else:
#                 j += 1
#         return i
#
#
# a = Solution()
# s = [1,2,3]
# g = [1,1]
# print(a.findContentChildren(s,g))



# class Solution:
#     def findDisappearedNumbers(self, nums: list[int]) -> list[int]:
#         n = len(nums)
#         m = list()
#         nums.sort()
#         for i in range(n):
#             if not nums.count(i+1):
#                 m.append(i+1)
#         return m
#
# a = Solution()
# nums = [4,3,2,7,8,2,3,1]
# print(a.findDisappearedNumbers(nums))



# class Solution:
#     def sampleStats(self, count: list[int]) -> list[float]:
#         num,counts = [],0
#         max1,max2 = 0,0
#         for i in range(len(count)):
#             if count[i] != 0:
#                 counts += i * count[i]
#                 if count[i] >= max1:
#                     max2 = i
#                     max1 = count[i]
#                 for j in range(count[i]):
#                     num.append(i)
#         if len(num) % 2 == 1:
#             mid = num[(len(num)-1) // 2]
#         else:
#             mid = (num[(len(num)) // 2] + num[(len(num)) // 2 - 1]) / 2
#         return [float(x) for x in [num[0],num[len(num) - 1],counts / len(num),mid,max1]]
#
#
# a = Solution()
# count = [0,1,3,4,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
# print(a.sampleStats(count))



# import heapq
#
# class Solution:
#     def minGroups(self, intervals: list[list[int]]) -> int:
#         intervals.sort(key=lambda p: p[0])
#         h = []
#         for left, right in intervals:
#             if h and left > h[0]: heapq.heapreplace(h, right)
#             else: heapq.heappush(h, right)
#         return len(h)
#
# a = Solution()
# intervals = [[5,10],[6,8],[1,5],[2,3],[1,10]]
# print(a.minGroups(intervals))



# class Solution:
#     def findJudge(self, n: int, trust: list[list[int]]) -> int:
#         num,nums = [0]*n,[0]*n
#         for i,j in trust:
#             num[i-1]+=1
#             nums[j-1]+=1
#         for k in range(n):
#             if num[k] == 0 and nums[k] == n-1:
#                 return k+1
#         return -1
#
#
# a = Solution()
# trust = [[1,2]]
# print(a.findJudge(2,trust))

# import collections
#
# class Solution:
#     def canFinish(self, numCourses: int, prerequisites: list[list[int]]) -> bool:
#         edges = collections.defaultdict(list)
#         # edges: defaultdict(<class 'list'>,{0: [3], 1: [3,4], 2: [4], 3: [5], 4: [5]}
#         indeg = [0] * numCourses
#         # indeg: [0,0,0,2,2,2]
#         for info in prerequisites:
#             edges[info[1]].append(info[0])
#             indeg[info[0]] += 1
#
#         q = collections.deque([u for u in range(numCourses) if indeg[u] == 0]) # [0,1,2]
#         visited = 0
#
#         while q:
#             visited += 1
#             u = q.popleft()
#             for v in edges[u]:
#                 indeg[v] -= 1
#                 if indeg[v] == 0:
#                     q.append(v)
#
#         return visited == numCourses

#
#
# a = Solution()
# numCourses, prerequisites = 6,[[3, 0], [3, 1], [4, 1], [4, 2], [5, 3], [5, 4]]
# print(a.canFinish(numCourses,prerequisites))


# class Solution:
#     desc = ("Gold Medal", "Silver Medal", "Bronze Medal")
#
#     def findRelativeRanks(self, score: list[int]) -> list[str]:
#         ans = [""] * len(score)
#         arr = sorted(enumerate(score), key=lambda x: -x[1])
#         for i, (idx, _) in enumerate(arr):
#             ans[idx] = self.desc[i] if i < 3 else str(i + 1)
#         return ans
#
# a = Solution()
# score = [10,3,8,9,4]
# print(a.findRelativeRanks(score))


# class Solution:
#     def climbStairs(self, n: int) -> int:
#         if n == 1:return 1
#         elif n == 2:return 2
#         else:
#             return self.climbStairs(n-1) + self.climbStairs(n-2)

# class Solution:
#     def climbStairs(self, n: int) -> int:
#         if n == 1:return 1
#         elif n == 2:return 2
#         else:
#             a,b = 1,2
#             c = 0
#             for i in range(n-2):
#                 c = a + b
#                 a = b
#                 b = c
#         return c


# class Solution:
#     def climbStairs(self, n: int) -> int:
#         if n <= 3:return n
#         return self.climbStairs(n//2) * self.climbStairs(n-n//2) + self.climbStairs(n//2-1) * self.climbStairs(n-n//2-1)
#
# a = Solution()
# n = int(input())
# print(a.climbStairs(n))

# import collections


# class Solution:
#     def sequenceReconstruction(self, nums: list[int], sequences: list[list[int]]) -> bool:
#
#         # 记录每个数的子结点
#         d = collections.defaultdict(set)
#         for seq in sequences:
#             for i in range(1, len(seq)):
#                 d[seq[i-1]].add(seq[i])
#
#       # 检查 nums 是否一条从头到尾的一条路径
#         for i in range(1, len(nums)):
#             if nums[i] not in d[nums[i-1]]:
#                 return False
#         return True
#

# import itertools,collections
#
# class Solution:
#     def sequenceReconstruction(self, nums: list[int], sequences: list[list[int]]) -> bool:
#         n = len(nums)
#         # [[],[],[],[]]
#         g = [[] for _ in range(n)]
#         inDeg = [0] * n
#         for sequence in sequences:
#             # 假如是[3,2,4],第一轮:x = 3,y = 2,第二轮:x = 2,y = 4
#             for x, y in itertools.pairwise(sequence):
#                 g[x - 1].append(y - 1)
#                 inDeg[y - 1] += 1
#
#         q = collections.deque([i for i, d in enumerate(inDeg) if d == 0])
#         while q:
#             if len(q) > 1:
#                 return False
#             x = q.popleft()
#             for y in g[x]:
#                 inDeg[y] -= 1
#                 if inDeg[y] == 0:
#                     q.append(y)
#         return True
#
#
#
# a = Solution()
# nums,sequences = [1,2,3,4],[[1,3],[3,2,4]]
# print(a.sequenceReconstruction(nums,sequences))


# class Solution:
#     def findPoisonedDuration(self, timeSeries: list[int], duration: int) -> int:
#         count = 0
#         for i in range(1,len(timeSeries)):
#             if timeSeries[i] - timeSeries[i-1] < duration:
#                 count += timeSeries[i] - timeSeries[i-1]
#             else:
#                 count += duration
#         count += duration
#         return count
#
# a = Solution()
# t = [1,2]
# print(a.findPoisonedDuration(t,2))
import collections
import heapq
from typing import List


# class Solution:
#     def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
#
#         def dfs(candidates, begin, size, path, res, target):
#             if target < 0:
#                 return
#             if target == 0:
#                 res.append(path)
#                 return
#
#             for index in range(begin, size):
#                 dfs(candidates, index, size, path + [candidates[index]], res, target - candidates[index])
#
#         size = len(candidates)
#         if size == 0:
#             return []
#         path = []
#         res = []
#         dfs(candidates, 0, size, path, res, target)
#         return res


from typing import List
# class Solution:
#     def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
#
#         def dfs(candidates, begin, size, path, res, target):
#             if target == 0:
#                 res.append(path)
#                 return
#
#             for index in range(begin, size):
#                 residue = target - candidates[index]
#                 if residue < 0:
#                     break
#
#                 dfs(candidates, index, size, path + [candidates[index]], res, residue)
#
#         size = len(candidates)
#         if size == 0:
#             return []
#         candidates.sort()
#         path = []
#         res = []
#         dfs(candidates, 0, size, path, res, target)
#         return res

#
#
# a = Solution()
# candidates,target = [2,3,6,7],7
# print(a.combinationSum(candidates,target))


# class Solution:
#     def minCostClimbingStairs(self, cost: List[int]) -> int:
#         n = len(cost)
#         dp = [0] * (n + 1)
#         for i in range(2, n + 1):
#             dp[i] = min(dp[i - 1] + cost[i - 1], dp[i - 2] + cost[i - 2])
#         return dp[n]
#
#
#
#
# a = Solution()
# cost = [10,15,20]
# print(a.minCostClimbingStairs(cost))



# class Solution:
#     def divisorGame(self, n: int) -> bool:
#         num = [0] * (n+5)
#         num[2] = 1
#         for i in range(3,n+1):
#             for j in range(1,i):
#                 if i % j == 0 and num[i-j] == 0:
#                     num[i] = 1
#                     break
#         return num[n] == 1
#
# a = Solution()
# n = int(input())
# print(a.divisorGame(n))


# class Solution:
#     def tribonacci(self, n: int) -> int:
#         if n == 0:return 0
#         if n <= 2:return 1
#         else:
#             a,b = 0,1
#             c = 1
#             for i in range(2,n):
#                 m = a + b + c
#                 a = b
#                 b = c
#                 c = m
#         return c
#
# a = Solution()
# n = int(input())
# print(a.tribonacci(n))


# class KthLargest:
#
#     def __init__(self, k: int, nums: List[int]):
#         self.k = k
#         self.que = nums
#         heapq.heapify(self.que)
#
#     def add(self, val: int) -> int:
#         heapq.heappush(self.que,val)
#         while len(self.que) > self.k:
#             heapq.heappop(self.que)
#         return self.que[0]
#
# k = 3
# nums = [4,5,8,2]
# a = KthLargest(k,nums)
# print(a.add(3))
# print(a.add(5))
# print(a.add(10))
# print(a.add(9))
# print(a.add(4))



# class Solution:
#     def addStrings(self, num1: str, num2: str) -> str:
#         i,j = len(num1)-1,len(num2)-1
#         add = 0
#         s = ""
#         while i >= 0 or j >= 0 or add != 0:
#             x = int(num1[i]) if i >= 0 else 0
#             y = int(num2[j]) if j >= 0 else 0
#             n = x + y + add
#             add = n // 10
#             s = str(n % 10) + s
#             i -= 1
#             j -= 1
#         return s
#
# a = Solution()
# s,t = '123','459'
# print(a.addStrings(s,t))



# class Solution:
#     def distributeCandies(self, candyType: List[int]) -> int:
#          n = len(candyType) // 2
#          s = set(candyType)
#          return n if n < len(s) else len(s)
#
# a = Solution()
# num = [1,1,21,1,1,1,1]
# print(a.distributeCandies(num))

# import collections
#
# class Solution:
#     def findLHS(self, nums: List[int]) -> int:
#         nums.sort()
#         res, begin = 0, 0
#         for end in range(len(nums)):
#             while nums[end] - nums[begin] > 1:
#                 begin += 1
#             if nums[end] - nums[begin] == 1:
#                 res = max(res, end - begin + 1)
#         return res
#
# a = Solution()
# nums = [1,2,3,3,1,-14,13,4]
# print(a.findLHS(nums))
import numpy


# class Solution:
#
#     base, count = 0, 0
#     maxCount = 0
#     num = []
#
#     def updata(self,x: int):
#         if x == self.base:
#             self.count += 1
#         else:
#             self.count = 1
#             self.base = x
#         if self.count == self.maxCount:
#             num.append(self.base)
#         if self.count > self.maxCount:
#             self.maxCount = self.count
#             del num
#             num.append(self.base)
#
#     def find(self,root):
#         if not root:
#             return
#         self.find(root.left)
#         updata(root.val)
#         self.find(root.right)
#
#
#     def findMode(self, root: Optional[TreeNode]) -> List[int]:
#         self.find(root)
#         return num

# class Solution:
#     def matrixReshape(self, mat: List[List[int]], r: int, c: int) -> List[List[int]]:
#         mats = sum(mat,[])
#         if len(mats) != r * c:
#             return mat
#         num = [[0] * c for _ in range(r)]
#         f = 0
#         for i in range(r):
#             for j in range(c):
#                 num[i][j] = mats[f]
#                 f += 1
#         return num
#
#
#
#
# a = Solution()
# mat = [[1,2,3],[4,5,6]]
# r,c = 1,6
# print(a.matrixReshape(mat,r,c))


# class Solution:
#     def threeSum(self, nums: List[int]) -> List[List[int]]:
#         n = len(nums)
#         nums.sort()
#         ans = list()
#
#         # 枚举 a
#         for first in range(n):
#             # 需要和上一次枚举的数不相同
#             if first > 0 and nums[first] == nums[first - 1]:
#                 continue
#             # c 对应的指针初始指向数组的最右端
#             third = n - 1
#             target = -nums[first]
#             # 枚举 b
#             for second in range(first + 1, n):
#                 # 需要和上一次枚举的数不相同
#                 if second > first + 1 and nums[second] == nums[second - 1]:
#                     continue
#                 # 需要保证 b 的指针在 c 的指针的左侧
#                 while second < third and nums[second] + nums[third] > target:
#                     third -= 1
#                 # 如果指针重合，随着 b 后续的增加
#                 # 就不会有满足 a+b+c=0 并且 b<c 的 c 了，可以退出循环
#                 if second == third:
#                     break
#                 if nums[second] + nums[third] == target:
#                     ans.append([nums[first], nums[second], nums[third]])
#
#         return ans
#
#
# a = Solution()
# nums = [-1,0,1,2,-1,-4]
# print(a.threeSum(nums))



# class Solution:
#     def Count(self,num: List[List[int]],m,n):
#         for i in range(m):
#             for j in range(n):
#                 num[i][j] += 1
#
#     def maxCount(self, m: int, n: int, ops: List[List[int]]) -> int:
#         count,num = 0,[[0] * n for _ in range(m)]
#         for i,j in ops:
#             if i == m and j == n:
#                 continue
#             else:
#                 self.Count(num,i,j)
#         num = sum(num,[])
#         return num.count(max(num))


# class Solution:
#     def maxCount(self, m: int, n: int, ops: List[List[int]]) -> int:
#         mina, minb = m, n
#         for a, b in ops:
#             mina = min(mina, a)
#             minb = min(minb, b)
#         return mina * minb
#
#
# a = Solution()
# m,n = 3,3
# ops = [[2,1],[2,2],[3,3],[2,3],[3,1]]
# print(a.maxCount(m,n,ops))



# class Solution:
#     def findRestaurant(self, list1: List[str], list2: List[str]) -> List[str]:
#         index = {s: i for i, s in enumerate(list1)}
#         ans = []
#         indexSum = float('inf')
#         for i, s in enumerate(list2):
#             if s in index:
#                 j = index[s]
#                 if i + j < indexSum:
#                     indexSum = i + j
#                     ans = [s]
#                     continue
#                 elif i + j == indexSum:
#                     ans.append(s)
#         return ans
#
#
#
# a = Solution()
# list1 = ["Shogun", "KFC", "Burger King", ]
# list2 = ["KFC", "Shogun", "Burger King"]
# print(a.findRestaurant(list1,list2))



# class Solution(object):
#     def longestIncreasingPath(self, matrix):
#         if not matrix or not matrix[0]:
#             return 0
#         m, n = len(matrix), len(matrix[0])
#         lst = []
#         for i in range(m):
#             for j in range(n):
#                 lst.append((matrix[i][j], i, j))
#         lst.sort()
#         dp = [[0 for _ in range(n)] for _ in range(m)]
#         for num, i, j in lst:
#             dp[i][j] = 1
#             for di, dj in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
#                 r, c = i + di, j + dj
#                 if 0 <= r < m and 0 <= c < n:
#                     if matrix[i][j] > matrix[r][c]:
#                         dp[i][j] = max(dp[i][j], 1 + dp[r][c])
#         return max([dp[i][j] for i in range(m) for j in range(n)])
#
#
# a = Solution()
# matrix = [[9,9,4],[6,6,8],[2,1,1]]
# print(a.longestIncreasingPath(matrix))


# class Solution:
#     def gameOfLife(self, board: List[List[int]]) -> None:
#         if not board or not board[0]:
#             return
#         m, n = len(board), len(board[0])
#         lst = []
#         count, count1 = 0, 0
#         for i in range(m):
#             for j in range(n):
#                 lst.append((board[i][j], i, j))
#         dp = []
#         for num, i, j in lst:
#             for di, dj in [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]:
#                 r, c = i + di, j + dj
#                 if 0 <= r < m and 0 <= c < n:
#                     if board[r][c] == 1:
#                         count += 1
#             if count == 3 and num == 0:
#                 dp.append(((i, j, 1)))
#             if count < 2 and num == 1:
#                 dp.append(((i, j, 0)))
#             if count > 3 and num == 1:
#                 dp.append(((i, j, 0)))
#             count = 0
#
#         for i, j, k in dp:
#             if k == 1:
#                 board[i][j] = 1
#             else:
#                 board[i][j] = 0
#
# a = Solution()
# board = [[1,1],[1,0]]
# print(a.gameOfLife(board))

# import math
# class Solution:
#     def leftRigthDifference(self, nums: List[int]) -> List[int]:
#         num, n = [], len(nums)
#         s, t = [0] * n, [0] * n
#         for i in range(n - 1):
#             s[i + 1] = sum(nums[:i + 1])
#             t[i] = sum(nums[n-1:i:-1])
#         for i in range(n):
#             num.append(int(math.fabs(s[i] - t[i])))
#         return num
#
# a = Solution()
# nums = [10,4,8,3]
# print(a.leftRigthDifference(nums))



# class Solution:
#     def maxNumOfMarkedIndices(self, nums: List[int]) -> int:
#         n = len(nums)
#         nums.sort()
#         l, r = 0, len(nums) // 2
#         while l <= r:
#             m = (l + r) // 2
#             flag = True
#             for i in range(m):
#                 if nums[i] * 2 > nums[n-m+i]:
#                     flag = False
#             if flag: l = m + 1
#             else: r = m - 1
#
#         return r * 2
#
#
# a = Solution()
# nums = [9,2,5,4]
# print(a.maxNumOfMarkedIndices(nums))




# class Solution:
#     def movesToMakeZigzag(self, nums: List[int]) -> int:
#         def help(pos: int) -> int:
#             res = 0
#             for i in range(pos, len(nums), 2):
#                 a = 0
#                 if i - 1 >= 0:
#                     a = max(a, nums[i] - nums[i - 1] + 1)
#                 if i + 1 < len(nums):
#                     a = max(a, nums[i] - nums[i + 1] + 1)
#                 res += a
#             return res
#
#         return min(help(0), help(1))
#
#
# a = Solution()
# nums = [9,6,1,6,2]
# print(a.movesToMakeZigzag(nums))

# class Solution:
#     def mergeSimilarItems(self, items1: List[List[int]], items2: List[List[int]]) -> List[List[int]]:
#         map = collections.Counter()
#         for a, b in items1:
#             map[a] += b
#         for a, b in items2:
#             map[a] += b
#         return sorted([a, b] for a, b in map.items())
#
# a = Solution()
# items1, items2 = [[1,1],[4,5],[3,8]], [[3,1],[1,5]]
# print(a.mergeSimilarItems(items1,items2))



# class Solution:
#     def printBin(self, num: float) -> str:
#         s = "0."
#         for i in range(1,20):
#             if num >= 2 ** (-i):
#                 num -= 2 ** (-i)
#                 s = s + '1'
#             elif num == 0:
#                 return s
#             else:
#                 s = s + '0'
#         return "ERROR"
#
#
# a = Solution()
# num = 0.
# print(a.printBin(num))


# class ParkingSystem:
#
#     def __init__(self, big: int, medium: int, small: int):
#         self.big = big
#         self.medium = medium
#         self.small = small
#
#     def addCar(self, carType: int) -> bool:
#         if carType == 1:
#             if self.big > 0:
#                 self.big -= 1
#                 return True
#             else:
#                 return False
#         if carType == 2:
#             if self.medium > 0:
#                 self.medium -= 1
#                 return True
#             else:
#                 return False
#         if carType == 3:
#             if self.small > 0:
#                 self.small -= 1
#                 return True
#             else:
#                 return False
#
# a = ParkingSystem(1,1,0)
# print(a.addCar(1))
# print(a.addCar(2))
# print(a.addCar(3))
# print(a.addCar(1))


# class Solution:
#     def getFolderNames(self, names: List[str]) -> List[str]:
#         ans = []
#         index = {}
#         for name in names:
#             if name not in index:
#                 ans.append(name)
#                 index[name] = 1
#             else:
#                 k = index[name]
#                 while name + '(' + str(k) + ')' in index:
#                     k += 1
#                 t = name + '(' + str(k) + ')'
#                 ans.append(t)
#                 index[name] = k + 1
#                 index[t] = 1
#         return ans
#
#
# a = Solution()
# names = ["gta","gta(1)","gta","avalon"]
# print(a.getFolderNames(names))


# class Solution:
#     def lengthOfLongestSubstring(self, s: str) -> int:
#         if not s:return 0
#         left = 0
#         lookup = set()
#         n = len(s)
#         max_len = 0
#         cur_len = 0
#         for i in range(n):
#             cur_len += 1
#             while s[i] in lookup:
#                 lookup.remove(s[left])
#                 left += 1
#                 cur_len -= 1
#             if cur_len > max_len:max_len = cur_len
#             lookup.add(s[i])
#         return max_len
# 
# a = Solution()
# s = "abccbabb"
# print(a.lengthOfLongestSubstring(s))


# class Solution:
#     def longestCommonSubsequence(self, text1: str, text2: str) -> int:
#         m,n = len(text1),len(text2)
#         dp = [[0] * (n+1) for i in range(m+1)]
#         for i in range(1,m+1):
#             for j in range(1,n+1):
#                 if text1[i-1] == text2[j-1]:
#                     dp[i][j] = dp[i-1][j-1] + 1
#                 else:
#                     dp[i][j] = max(dp[i][j-1],dp[i-1][j])
#         les = dp[m][n]
#         return les
#
#
# a = Solution()
# word1 = "park"
# word2 = "spake"
# print(a.longestCommonSubsequence(word1,word2))



# class Solution:
#     def canPlaceFlowers(self, flowerbed: List[int], n: int) -> bool:
#         i,n = 0,len(flowerbed)
#         for i in range(n):
#             if flowerbed[i] == 0 and (i == 0 or flowerbed[i-1] == 0) and (i == n-1 or flowerbed[i+1] == 0):
#                 n -= 1
#                 flowerbed[i] = 1
#         return n <= 0
#
#
# a = Solution()
# flowerbed = [1,0,0,0,1]
# print(a.canPlaceFlowers(flowerbed,1))





# from collections import Counter
#
# class Solution:
#     def findErrorNums(self, nums):
#         ln = len(nums)
#         dic = Counter(nums)
#         repeat = lose = -1
#         for i in range(1, ln + 1):
#             tmp = dic.get(i, 0)
#             if tmp == 0:
#                 lose = i
#             elif tmp == 2:
#                 repeat = i
#         return [repeat, lose]
#
#
# a = Solution()
# nums = [3,2,2]
# print(a.findErrorNums(nums))


# class Solution:
#     def maxValue(self, grid: List[List[int]]) -> int:
#         m, n = len(grid), len(grid[0])
#         f = [[0] * n for _ in range(m)]
#
#         for i in range(m):
#             for j in range(n):
#                 if i > 0:
#                     f[i][j] = max(f[i][j], f[i - 1][j])
#                 if j > 0:
#                     f[i][j] = max(f[i][j], f[i][j - 1])
#                 f[i][j] += grid[i][j]
#
#         return f[m - 1][n - 1]
#
#
#
#
# a = Solution()
# grid = [[1,3,1],[1,5,1],[4,2,1]]
# print(a.maxValue(grid))


# class Solution:
#     def getSumAbsoluteDifferences(self, nums: List[int]) -> List[int]:
#         n = len(nums)
#         num = [0] * n
#         num[0] = sum(nums) - nums[0] * n
#         for i in range(1, n):
#             d = nums[i] - nums[i - 1]
#             num[i] = ans[i - 1] - (n - i * 2) * d
#         return num
#
#
# a = Solution()
# nums = [1,4,6,8,10]
# print(a.getSumAbsoluteDifferences(nums))


# class Solution:
#     def findTargetSumWays(self, nums: List[int], target: int) -> int:
#
#         cache = {}  # 记忆化单元
#
#         # @functools.cache  # Python functools自带记忆化单元【启用后可省去自定义cache单元】
#         def dfs(i, summ, t):
#             '''summ: 前i个元素的表达式之和; t: 目标值'''
#             if (i, summ) in cache:  # 记忆化：已存在，直接返回
#                 return cache[(i, summ)]
#
#             if i == len(nums):  # 遍历完了全部的元素，递归中止
#                 if summ == t:  # 找到了一个满足要求的组合
#                     cache[(i, summ)] = 1
#                 else:
#                     cache[(i, summ)] = 0
#                 return cache[(i, summ)]
#
#             pos_cnt = dfs(i + 1, summ + nums[i], t)  # nums[i]前面添加'+'号
#             neg_cnt = dfs(i + 1, summ - nums[i], t)  # nums[i]前面添加'-'号
#             cache[(i, summ)] = pos_cnt + neg_cnt  # 以上两种情况的组合数之和
#             return cache[(i, summ)]
#
#         return dfs(0, 0, target)

# class Solution:
#     def minimumRecolors(self, blocks: str, k: int) -> int:
#         n,cur = len(blocks),0
#         min1,count = blocks.count("W"),0
#         for i in range(n):
#             if blocks[i] == "W":
#                 count += 1
#             if i - cur + 1 == k:
#                 min1 = min(count,min1)
#                 if blocks[cur] == "W":
#                     count -= 1
#                 cur += 1
#         return min1
#
#
# a = Solution()
# blocks = "WBBWWBBWBW"
# k = 7
# print(a.minimumRecolors(blocks,k))



# import pandas as pd
#
# pd.set_option('display.unicode.ambiguous_as_wide', True)
# pd.set_option('display.unicode.east_asian_width', True)
# pd.set_option('display.width', 100) # 设置打印宽度(**重要**)
#
# d = {"姓名": ["刘文涛","王宇翔","田思雨","徐丽娜","丁文彬"],
#      "统计学": [68,85,74,88,63],
#     "经济学": [85,91,74,100,82],
#     "数学": [84,63,61,49,89]}
# table_1 = pd.DataFrame(d)
# print(table_1)


# class Solution:
#     def minMoves2(self, nums: List[int]) -> int:
#         nums.sort()
#         n = len(nums)
#         s = sum(nums) - nums[0] * n
#         min1 = s
#         for i in range(1, n):
#             d = nums[i] - nums[i - 1]
#             s = s - (n - i * 2) * d
#             min1 = min(s,min1)
#         return min1
#
#
# a = Solution()
# nums = [1,10,2,9]
# print(a.minMoves2(nums))



# class Solution:
#     def closetTarget(self, words: List[str], target: str, startIndex: int) -> int:
#         s = t = startIndex
#         for i in range(len(words)):
#             if words[s] == target:
#                 return i
#             elif words[t] == target:
#                 return i
#             elif s == len(words)-1:
#                 s = -1
#             elif t == 0:
#                 t = len(words)
#             s += 1
#             t -= 1
#         return -1
#
#
# a = Solution()
# words = ["a","b","leetcode"]
# target = "leetcode"
# startIndex = 0
# print(a.closetTarget(words,target,startIndex))



# class Solution:
#     def minSubarray(self, nums: List[int], p: int) -> int:
#         x = sum(nums) % p
#         if x == 0:
#             return 0
#         y = 0
#         index = {0: -1}
#         ans = len(nums)
#         for i, v in enumerate(nums):
#             y = (y + v) % p
#             if (y - x) % p in index:
#                 ans = min(ans, i - index[(y - x) % p])
#             index[y] = i
#         return ans if ans < len(nums) else -1
#
# a = Solution()
# nums = [3,1,4,2]
# p = 6
# print(a.minSubarray(nums,p))

# import collections
# class Solution:
#     def uncommonFromSentences(self, s1: str, s2: str) -> List[str]:
#         freq = collections.Counter(s1.split()) + collections.Counter(s2.split())
#
#         ans = list()
#         for word, occ in freq.items():
#             if occ == 1:
#                 ans.append(word)
#
#         return ans
#
#
# a = Solution()
# s1 = "this apple is sweet"
# s2 = "this apple is sour"
# print(a.uncommonFromSentences(s1,s2))


# class Solution:
#     def sumOfNumberAndReverse(self, num: int) -> bool:
#         def reverse(k:int)->int:
#             if k < 10:
#                 return k
#             else:
#                 str_x = str(k)
#                 str_x = str_x[::-1]
#                 k = int(str_x)
#                 return k
#
#         for i in range(num//2,num):
#             if i + reverse(i) == num:
#                 return True
#         return False
#
#
# a = Solution()
# num = 181
# print(a.sumOfNumberAndReverse(num))



# class Solution:
#     def imageSmoother(self, img: List[List[int]]) -> List[List[int]]:
#         m, n = len(img), len(img[0])
#         ans = [[0] * n for _ in range(m)]
#         for i in range(m):
#             for j in range(n):
#                 tot, num = 0, 0
#                 for x in range(max(i - 1, 0), min(i + 2, m)):
#                     for y in range(max(j - 1, 0), min(j + 2, n)):
#                         tot += img[x][y]
#                         num += 1
#                 ans[i][j] = tot // num
#         return ans
#
#
# a = Solution()
# img = [[1,2,3,4,5],[6,7,8,9,10],[11,12,13,14,15],[16,17,18,19,20],[21,22,23,24,25]]
# print(a.imageSmoother(img))


# class Solution:
#     def backspaceCompare(self, s: str, t: str) -> bool:
#         num = []
#         for i in range(len(s)):
#             if s[i] != '#':
#                 num.append(s[i])
#             else:
#                 if num:
#                     num.pop()
#         nums = []
#         for i in range(len(t)):
#             if t[i] != '#':
#                 nums.append(s[i])
#             else:
#                 if nums:
#                     nums.pop()
#         return num == nums
#
# a = Solution()
# s = "ab##"
# t = "a##c#"
# print(a.backspaceCompare(s,t))


# class Solution:
#     def compress(self, chars: list[str]) -> int:
#         def reverse(left,right):
#             while left < right:
#                 chars[left], chars[right] = chars[right], chars[left]
#                 left += 1
#                 right -= 1
#
#         n = len(chars)
#         write = left = 0
#         for i in range(n):
#             if i == n-1 or chars[i] != chars[i+1]:
#                 chars[write] = chars[i]
#                 write += 1
#                 num = i - left + 1
#                 if num > 1:
#                     s = write
#                     while num > 0:
#                         chars[write] = str(num % 10)
#                         num //= 10
#                         write += 1
#                     reverse(s,write-1)
#                 left = i + 1
#         return write
#
#
#
# a = Solution()
# chars = ["a","b","b","b","b","b","b","b","b","b","b","b","b"]
# print(a.compress(chars))


# class Solution:
#     def removeSubfolders(self, folder: List[str]) -> List[str]:
#         folder.sort()
#         ans = [folder[0]]
#         for i in range(1, len(folder)):
#             if not ((pre := len(ans[-1])) < len(folder[i]) and ans[-1] == folder[i][:pre] and folder[i][pre] == "/"):
#                 ans.append(folder[i])
#         return ans
#
#
#
# a = Solution()
# folder = ["/a","/a/b","/c/d","/c/d/e","/c/f"]
# print(a.removeSubfolders(folder))


# class Solution:
#     def findLengthOfLCIS(self, nums: List[int]) -> int:
#         left = count = 0
#         n = len(nums)
#         for i in range(n):
#             if i == n-1 or nums[i+1] <= nums[i]:
#                 count = max(count,i-left+1)
#                 left = i+1
#         return count
#
#
# a = Solution()
# nums = [1,3,5,4,7,8,9]
# print(a.findLengthOfLCIS(nums))


# class Solution:
#     def restoreMatrix(self, rowSum: List[int], colSum: List[int]) -> List[List[int]]:
#         n, m = len(rowSum), len(colSum)
#         matrix = [[0] * m for _ in range(n)]
#         i = j = 0
#         while i < n and j < m:
#             v = min(rowSum[i], colSum[j])
#             matrix[i][j] = v
#             rowSum[i] -= v
#             colSum[j] -= v
#             if rowSum[i] == 0:
#                 i += 1
#             if colSum[j] == 0:
#                 j += 1
#         return matrix
#
# rowSum = [3,8]
# colSum = [4,7]
# a = Solution()
# print(a.restoreMatrix(rowSum,colSum))


# class Solution:
#     def balancedStringSplit(self, s: str) -> int:
#         l = r = 0
#         count = 0
#         for i in s:
#             if i == 'L':
#                 l += 1
#             if i == 'R':
#                 r += 1
#             if l == r:
#                 count += 1
#         return count
#
# a = Solution()
# s = "LLLLRRRR"
# print(a.balancedStringSplit(s))


# class Solution:
#     def maxAscendingSum(self, nums: List[int]) -> int:
#         Max = count = 0
#         for i in range(len(nums)-1):
#             count += nums[i]
#             if nums[i] >= nums[i+1]:
#                 Max = max(count, Max)
#                 count = 0
#         count += nums[-1]
#         return max(count,Max)
#
#
# a = Solution()
# nums = [10,20,30,5,10,50]
# print(a.maxAscendingSum(nums))


# class Solution:
#     def maximalNetworkRank(self, n: int, roads: List[List[int]]) -> int:
#         nums = [[0] * n for i in range(n)]
#         num = [0] * n
#         for i, j in roads:
#             nums[i][j] = 1
#             num[i] += 1
#             num[j] += 1
#         count = 0
#         for i in range(n):
#             for j in range(i+1,n):
#                 m = num[i] + num[j] - nums[i][j]
#                 count = max(m,count)
#         return count
#
#
#
# roads = [[0,1],[1,2],[2,3],[2,4],[5,6],[5,7]]
# n = 8
# a = Solution()
# print(a.maximalNetworkRank(n,roads))

# class Solution:
#     def kthFactor(self, n: int, k: int) -> int:
#         count = 0
#         for i in range(1,n+1):
#             if n % i == 0:
#                 count += 1
#             if count == k:
#                 return i
#         return -1
#
# a = Solution()
# print(a.kthFactor(4,4))


# class Solution:
#     def numsSameConsecDiff(self, n: int, k: int) -> List[int]:
#         def c_k(num,n,k):
#             count = num % 10
#             num //= 10
#             for i in range(n-1):
#                 m = num % 10
#                 if m - count == k or count - m == k:
#                     count = m
#                 else:
#                     return False
#                 num //= 10
#             return True
#         ans = []
#         for i in range(10**(n-1),10**n):
#             if c_k(i,n,k):
#                 ans.append(i)
#         return ans
#
# a = Solution()
# print(a.numsSameConsecDiff(2,1))

# class Solution:
#     def numsSameConsecDiff(self, n: int, k: int) -> List[int]:
#         from collections import deque
#         # 初始化队列：2 <= n <= 9
#         q = deque(range(1, 10))
#         while n > 1:
#             for _ in range(len(q)):
#                 u = q.popleft()
#                 for v in {u % 10 - k, u % 10 + k}:  # 不用集合就判断k=0
#                     if 0 <= v <= 9:
#                         q.append(u * 10 + v)
#             n -= 1  # 层数
#         return list(q)  # 最后一层队列元素
#
# a = Solution()
# print(a.numsSameConsecDiff(3,7))

# import collections
# class Solution:
#     def minNumBooths(self, demand: List[str]) -> int:
#         res = collections.Counter()
#         for d in demand:
#             tmp = collections.Counter(d)
#             for key in tmp.keys():
#                 res[key] = res[key] if res[key] >= tmp[key] else tmp[key]
#         return sum(res.values())
#
#
# a = Solution()
# demand = ["acd","bed","accd"]
# print(a.minNumBooths(demand))


# class Solution:
#     def minimumRemoval(self, beans: List[int]) -> int:
#         beans.sort()
#         n,max1 = len(beans),0
#         for i in range(n):
#             count = beans[i] * (n-i)
#             max1 = max(count,max1)
#         return sum(beans)-max1
#
# a = Solution()
# beans = [2,10,3,2]
# print(a.minimumRemoval(beans))


# class Solution:
#     def numDecodings(self, s: str) -> int:
#         n = len(s)
#         f = [1] + [0] * n
#         for i in range(1, n + 1):
#             if s[i - 1] != '0':
#                 f[i] += f[i - 1]
#             if i > 1 and s[i - 2] != '0' and int(s[i-2:i]) <= 26:
#                 f[i] += f[i - 2]
#         return f[n]
#
#
# a = Solution()
# s = "11106"
# print(a.numDecodings(s))

# class Solution:
#     def longestWord(self, words: List[str]) -> str:
#         words.sort(key=lambda x: (-len(x), x), reverse=True)
#         longest = ""
#         candidates = {""}
#         for word in words:
#             if word[:-1] in candidates:
#                 longest = word
#                 candidates.add(word)
#         return longest
#
#
# a = Solution()
# words = ["a", "banana", "app", "appl", "ap", "apply", "apple",'b','ba','bax']
# print(a.longestWord(words))

# class NumMatrix:
#
#     def __init__(self, matrix: List[List[int]]):
#         m, n = len(matrix), (len(matrix[0]) if matrix else 0)
#         self.sums = [[0] * (n + 1) for _ in range(m)]
#         _sums = self.sums
#
#         for i in range(m):
#             for j in range(n):
#                 _sums[i][j + 1] = _sums[i][j] + matrix[i][j]
#
#     def sumRegion(self, row1: int, col1: int, row2: int, col2: int) -> int:
#         _sums = self.sums
#
#         total = sum(_sums[i][col2 + 1] - _sums[i][col1] for i in range(row1, row2 + 1))
#         return total
#
#
#
# matrix = [[3, 0, 1, 4, 2], [5, 6, 3, 2, 1], [1, 2, 0, 1, 5], [4, 1, 0, 1, 7], [1, 0, 3, 0, 5]]
# a = NumMatrix(matrix)
# print(a.sumRegion(2,1,4,3))
#


# class AnimalShelf:
#
#     def __init__(self):
#         self.num = []
#
#     def enqueue(self, animal: List[int]) -> None:
#         self.num.append(animal)
#
#     def dequeueAny(self) -> List[int]:
#         if len(self.num) == 0:return [-1,-1]
#         else:
#             return self.num.pop(0)
#
#     def dequeueDog(self) -> List[int]:
#         n,m = len(self.num),[]
#         for i in range(n):
#             if self.num[i][1] == 1:
#                 m = self.num[i]
#                 del self.num[i]
#                 return m
#         return [-1,-1]
#
#
#     def dequeueCat(self) -> List[int]:
#         n, m = len(self.num), []
#         for i in range(n):
#             if self.num[i][1] == 0:
#                 m = self.num[i]
#                 del self.num[i]
#                 return m
#         return [-1,-1]
#
# a = AnimalShelf()
# a.enqueue([0,0])
# a.enqueue([1,0])
# a.enqueue([2,1])
# print(a.dequeueAny())
# print(a.dequeueAny())
# print(a.dequeueAny())

# class Solution:
#     def checkPalindromeFormation(self, a: str, b: str) -> bool:
#         def rev(s):
#             return s[::-1]
#
#         if rev(a) == a or rev(b) == b:
#             return True
#         for i in range(len(a)):
#             s = a[:i + 1] + b[i+1:]
#             t = b[:i + 1] + a[i + 1:]
#             if rev(s) == s or rev(t) == t:
#                 return True
#         return False

# class Solution:
#     def checkPalindromeFormation(self, a: str, b: str) -> bool:
#         return self.checkConcatenation(a, b) or self.checkConcatenation(b, a)
#
#     def checkConcatenation(self, a: str, b: str) -> bool:
#         n = len(a)
#         left, right = 0, n - 1
#         while left < right and a[left] == b[right]:
#             left += 1
#             right -= 1
#         if left >= right:
#             return True
#         return self.checkSelfPalindrome(a, left, right) or self.checkSelfPalindrome(b, left, right)
#
#     def checkSelfPalindrome(self, a: str, left: int, right: int) -> bool:
#         while left < right and a[left] == a[right]:
#             left += 1
#             right -= 1
#         return left >= right
#
# a = Solution()
# s = "abdef"
# t = "fecab"
# print(a.checkPalindromeFormation(s,t))


# class Solution:
#     def minSwaps(self, nums: List[int]) -> int:
#         n = len(nums)
#         m = sum(nums)
#         if m == 0:return 0
#         cur = 0
#         for i in range(m):
#             cur += (1-nums[i])
#         ans = cur
#         for j in range(1,n):
#             if nums[j-1] == 0:
#                 cur -= 1
#             if nums[(j+m-1)%n] == 0:
#                 cur += 1
#             ans = min(ans,cur)
#         return ans
#
# a = Solution()
# nums = [0,1,1,1,0,0,1,1,0]
# print(a.minSwaps(nums))

# import itertools
# class Solution:
#     def largestTimeFromDigits(self, arr: List[int]) -> str:
#         ans = -1
#         for h1,h2,m1,m2 in itertools.permutations(arr):
#             hours = 10 * h1 + h2
#             mins = 10 * m1 + m2
#             time = 60 * hours + mins
#             if 0 <= hours < 24 and 0 <= mins < 60 and time > ans:
#                 ans = time
#         return "{:02}:{:02}".format(*divmod(ans, 60)) if ans >= 0 else ""
#
# a = Solution()
# arr = [0,0,0,0]
# print(a.largestTimeFromDigits(arr))


# class Solution:
#     def evenOddBit(self, n: int) -> List[int]:
#         num = [0,0]
#         i = 0
#         while n:
#             if n % 2 == 1:
#                 if i % 2 == 0:
#                     num[0] += 1
#                 else:
#                     num[1] += 1
#             n //= 2
#             i += 1
#         return num
#
# a = Solution()
# print(a.evenOddBit(1))


# class Solution:
#     def checkValidGrid(self, grid: List[List[int]]) -> bool:
#         n = len(grid)
#         count = 0
#         num = [0,0,grid[0][0]]
#         for i in range(n*n-1):
#             for x,y in [(1,2),(2,1),(-1,-2),(-1,2),(1,-2),(2,-1),(-2,-1),(-2,1)]:
#                 x1,y1 = num[0] + x,num[1] + y
#                 if 0 <= x1 < n and 0 <= y1 < n and grid[x1][y1] == num[2] + 1:
#                     num[0] = x1
#                     num[1] = y1
#                     num[2] += 1
#                     count += 1
#         return count == n*n-1
#
# a = Solution()
# grid = [[24,11,22,17,4],[21,16,5,12,9],[6,23,10,3,18],[15,20,1,8,13],[0,7,14,19,2]]
# print(a.checkValidGrid(grid))


# class Solution:
#     def beautifulSubsets(self, nums: List[int], k: int) -> int:
#         ans = -1  # 去掉空集
#         cnt = [0] * (max(nums) + k * 2)  # 用数组实现比哈希表更快
#         def dfs(i: int) -> None:
#             if i == len(nums):
#                 nonlocal ans
#                 ans += 1
#                 return
#             dfs(i + 1)  # 不选
#             x = nums[i]
#             if cnt[x - k] == 0 and cnt[x + k] == 0:
#                 cnt[x] += 1  # 选
#                 dfs(i + 1)
#                 cnt[x] -= 1  # 恢复现场
#         dfs(0)
#         return ans
#
#
# a = Solution()
# nums = [2,4,6,7,9]
# k = 2
# print(a.beautifulSubsets(nums,k))


# class Solution:
#     def checkValidGrid(self, grid: List[List[int]]) -> bool:
#         pos = [None] * (len(grid) ** 2)
#         for i, row in enumerate(grid):
#             for j, x in enumerate(row):
#                 pos[x] = (i,j)
#
#         if pos[0] != (0,0):
#             return False
#
#         for (i,j), (x,y) in pairwise(pos):
#             dx = abs(x - i)
#             dy = abs(y - i)
#             if (dx != 2 or dy != 1) and (dx != 1 or dy != 2):
#                 return False
#
#         return True
#
#
# a = Solution()
# grid = [[0,11,16,5,20],[17,4,19,10,15],[12,1,8,21,6],[3,18,23,14,9],[24,13,2,7,22]]
# print(a.checkValidGrid(grid))

# import math
# class Solution:
#     def numDupDigitsAtMostN(self, N: int) -> int:
#         limit, s = list(map(int, str(N + 1))), set()
#         n, res = len(limit), sum(9 * math.perm(9, i) for i in range(len(limit) - 1))
#         for i, x in enumerate(limit):
#             for y in range(i == 0, x):
#                 if y not in s:
#                     res += math.perm(9 - i, n - i - 1)
#             if x in s:
#                 break
#             s.add(x)
#         return N - res
#
# a = Solution()
# print(a.numDupDigitsAtMostN(256))


# class Solution:
#     def help(self, h1: List[int], h2: List[int], diff: int) -> int:
#         h = [0] * 7
#         for i in range(1, 7):
#             h[6 - i] += h1[i]
#             h[i - 1] += h2[i]
#         res = 0
#         for i in range(5, 0, -1):
#             if diff <= 0: break
#             t = min((diff + i - 1) // i, h[i])
#             res += t
#             diff -= t * i
#         return res
#
#     def minOperations(self, nums1: List[int], nums2: List[int]) -> int:
#         n, m = len(nums1), len(nums2)
#         if 6 * n < m or 6 * m < n:
#             return -1
#         cnt1 = [0] * 7
#         cnt2 = [0] * 7
#         diff = 0
#         for i in nums1:
#             cnt1[i] += 1
#             diff += i
#         for i in nums2:
#             cnt2[i] += 1
#             diff -= i
#         if diff == 0:
#             return 0
#         if diff > 0:
#             return self.help(cnt2, cnt1, diff)
#         return self.help(cnt1, cnt2, -diff)
#
#
#
# a = Solution()
# nums1 = [6,6]
# nums2 = [1]
# print(a.minOperations(nums1,nums2))



# class Solution:
#     def toHex(self, num: int) -> str:
#         CONV = "0123456789abcdef"
#         ans = []
#         # 32位2进制数，转换成16进制 -> 4个一组，一共八组
#         for _ in range(8):
#             ans.append(num%16)
#             num //= 16
#             if not num:
#                 break
#         return "".join(CONV[n] for n in ans[::-1])
#
#
#
# a = Solution()
# print(a.toHex(-2))


# class Solution:
#     def change(self, amount: int, coins: List[int]) -> int:
#
#         n = len(coins)
#         dp = [[0] * (amount + 1) for _ in range(n + 1)]  # 初始化
#         dp[0][0] = 1  # 合法的初始化
#
#         # 完全背包：优化后的状态转移
#         for i in range(1, n + 1):  # 第一层循环：遍历硬币
#             for j in range(amount + 1):  # 第二层循环：遍历背包
#                 if j < coins[i - 1]:  # 容量有限，无法选择第i个硬币
#                     dp[i][j] = dp[i - 1][j]
#                 else:  # 可选择第i个硬币
#                     dp[i][j] = dp[i - 1][j] + dp[i][j - coins[i - 1]]
#
#         return dp[n][amount]
#
#
#
# a = Solution()
# amount = 5
# coins = [1,2,5]
# print(a.change(amount,coins))


# class Solution:
#     def bestTeamScore(self, scores: List[int], ages: List[int]) -> int:
#         people = sorted(zip(scores, ages))
#         dp = [0] * len(scores)
#         ans = 0
#         for i in range(len(scores)):
#             for j in range(i):
#                 if people[i][1] >= people[j][1]:
#                     dp[i] = max(dp[i], dp[j])
#             dp[i] += people[i][0]
#             ans = max(ans, dp[i])
#         return ans
#
#
#
# a = Solution()
# scores = [4,5,6,5]
# ages = [2,1,2,1]
# print(a.bestTeamScore(scores,ages))

# import math
# class Solution:
#     def subarrayLCM(self, nums: List[int], k: int) -> int:
#         ans = 0
#         res = 1
#         for i in range(len(nums)):
#             for j in range(i,len(nums)):
#                 res = math.lcm(res,nums[j])
#                 if res == k:ans += 1
#                 elif math.lcm(res,k) != k:
#                     res = 1
#                     break
#             res = 1
#         return ans
#
#
# a = Solution()
# nums = [3,6,2,7,1]
# k = 6
# print(a.subarrayLCM(nums,k))

# # import collections
# class Solution:
#     def longestPalindrome(self, words: List[str]) -> int:
#         num = collections.Counter(words)
#         count = 0
#         flag = False
#         for i, j in num.items():
#             if i[0] == i[1]:
#                 if j % 2 == 1:
#                     count += (j - 1)
#                     flag = True
#                 else:
#                     count += j
#             elif i[::-1] in num:
#                 t = i[::-1]
#                 count += min(j,num[t]) * 2
#                 num[i] = 0
#                 num[t] = 0
#         return count * 2 + flag * 2
#
# a = Solution()
# words = ["lc","cl","gg"]
# print(a.longestPalindrome(words))

import collections
class Solution:
    def makeConnected(self, n: int, connections: List[List[int]]) -> int:
        if len(connections) < n - 1:
            return -1
        edges = collections.defaultdict(list)
        for i, j in connections:
            edges[i].append(j)
            edges[j].append(i)
        ans = 0
        seen = set()
        def dfs(u: int):
            seen.add(u)
            for v in edges[u]:
                if v not in seen:
                    dfs(v)
        for i in range(n):
            if i not in seen:
                dfs(i)
                ans += 1
        return ans - 1





a = Solution()
n = 4
connections = [[0,1],[0,2],[1,2]]
print(a.makeConnected(n,connections))

