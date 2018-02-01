#!/usr/bin/python

import unittest
from MergeIntersection import countIntersection

def verifyIntersectCount(u, v):
    count = 0
    vi_start, ui_start = 0, 0
    while (vi_start < len(v)) and (ui_start < len(u)):
        if u[ui_start] == v[vi_start]:
            count += 1
        comp1 = u[ui_start] >= v[vi_start]
        comp2 = u[ui_start] <= v[vi_start]
        ui_bound = (ui_start == len(u)-1)
        vi_bound = (vi_start == len(v)-1)
        if (comp1 and vi_start < len(v)-1) or ui_bound:
            vi_start += 1
        if (comp2 and ui_start < len(u)-1) or vi_bound:
            ui_start += 1
    return count


def printIntersectCount(u, v, threads):
    print("u: {}".format(u))
    print("v: {}".format(v))
    print("threads: {}".format(threads))
    count = countIntersection(u, v, threads)
    print("Intersect count: {}".format(count))
    print("\n")
    return count


class countIntersectionTest(unittest.TestCase):

    def testSmallest(self):
    	u, v = [1], [1] 
    	count = printIntersectCount(u, v, 1)
        self.assertEqual(count, verifyIntersectCount(u, v))

    def testSingleThread(self):
		u, v = [2, 4], [3, 4] 
		count = printIntersectCount(u, v, 2)
		self.assertEqual(count, verifyIntersectCount(u, v))
    def testTwoThreads(self):
		u, v = [2, 4], [3, 4] 
		count = printIntersectCount(u, v, 2)
		self.assertEqual(count, verifyIntersectCount(u, v))

    def testThreeThreads(self):
        u, v = [2, 4], [3, 4] 
        count = printIntersectCount(u, v, 3)
        self.assertEqual(count, verifyIntersectCount(u, v))

    def testMoreThreadsThanWork(self):
        u, v = [2, 4], [3, 4] 
        count = printIntersectCount(u, v, 4)
        self.assertEqual(count, verifyIntersectCount(u, v))

    def testTwoByTwoCase1(self):
        u, v = [1, 2], [1, 2] 
        count = printIntersectCount(u, v, 2)
        self.assertEqual(count, verifyIntersectCount(u, v))

    def testTwoByTwoCase2(self):
        u, v = [1, 3], [1, 2]
        count = printIntersectCount(u, v, 2)
        self.assertEqual(count, verifyIntersectCount(u, v))

    def testTwoByTwoCase3(self):
        u, v = [1, 4], [2, 4]
        count = printIntersectCount(u, v, 2)
        self.assertEqual(count, verifyIntersectCount(u, v))

    def testTwoByTwoCase4(self):
        u, v = [1, 3], [2, 3]
        count = printIntersectCount(u, v, 2)
        self.assertEqual(count, verifyIntersectCount(u, v))

    def testTwoByTwoCase5(self):
        u, v = [1, 2], [2, 3]
        count = printIntersectCount(u, v, 2)
        self.assertEqual(count, verifyIntersectCount(u, v))

    def testTwoByTwoCase6(self):
        u, v = [1, 2], [3, 4]
        count = printIntersectCount(u, v, 2)
        self.assertEqual(count, verifyIntersectCount(u, v))

    def test2x1(self):
        u, v = [2], [1, 2, 3] 
        count = printIntersectCount(u, v, 3)
        self.assertEqual(count, verifyIntersectCount(u, v))

    def test5x3(self):
        u, v = [2, 4, 8], [1, 2, 4, 7, 8] 
        count = printIntersectCount(u, v, 4)
        self.assertEqual(count, verifyIntersectCount(u, v))

# def testBinarySearch():
#     u, v = [2, 4, 8], [1, 2, 5, 7, 8] 
#     threads_per_intersect = 4
#     print(bSearch(u, v, 0, 1, 1, 0))
#     print(bSearch(u, v, 1, 2, 3, 0))
#     print(bSearch(u, v, 3, 2, 4, 1))
#     print("\n")

#     u, v = [6, 7, 8], [1, 2, 3, 4, 5] 
#     threads_per_intersect = 4
#     print(bSearch(u, v, 0, 1, 1, 0))
#     print(bSearch(u, v, 1, 2, 3, 0))
#     print(bSearch(u, v, 3, 2, 4, 1))
#     print("\n")

#     u, v = [1, 2, 3], [4, 5, 6, 7, 8] 
#     threads_per_intersect = 4
#     print(bSearch(u, v, 0, 1, 1, 0))
#     print(bSearch(u, v, 1, 2, 3, 0))
#     print(bSearch(u, v, 3, 2, 4, 1))
#     print("\n")


# def testCountIntersection():
#     u, v = [2, 4, 8], [1, 2, 5, 7, 8] 
#     printTestIntersectCount(u, v, 4)

#     # "smallest" test case
#     u2, v2 = [1], [1] 
#     printTestIntersectCount(u2, v2, 1)

#     # # threads == |comparisons (work-1)|
#     # u3, v3 = [2, 4], [3, 4] 
#     # printTestIntersectCount(u3, v3, 3)

#     # # threads == |work|
#     # u3, v3 = [2, 4], [3, 4] 
#     # printTestIntersectCount(u3, v3, 4)

#     # # threads > |work|
#     # u3, v3 = [2, 4], [3, 4] 
#     # printTestIntersectCount(u3, v3, 5)

#     # 2x2 with 1 thread test case
#     u4, v4 = [2, 3], [3, 4] 
#     printTestIntersectCount(u4, v4, 1)

#     # u1 < v1 = u2 < v2
#     u5, v5 = [2, 3], [3, 4]
#     printTestIntersectCount(u5, v5, 2)

#     # v1 < u1 < v2 < u2
#     u6, v6 = [2, 4], [3, 5] 
#     printTestIntersectCount(u6, v6, 2)

#     # v1 < u1 < u2 < v2
#     u7, v7 = [3, 4], [2, 5] 
#     printTestIntersectCount(u7, v7, 2)

#     # u1 < v1 < v1 < u2
#     u8, v8 = [2, 5], [3, 4] 
#     printTestIntersectCount(u8, v8, 2)

#     # u1 < u2 < v1 < v2
#     u9, v9 = [1, 2], [3, 4] 
#     printTestIntersectCount(u9, v9, 2)

#     # v1 < v2 < u1 < u2
#     u10, v10 = [3, 4], [1, 2] 
#     printTestIntersectCount(u10, v10, 2)

#     # u1 = v1 < u2 = v2
#     u11, v11 = [1, 2], [1, 2] 
#     printTestIntersectCount(u11, v11, 2)

if __name__ == '__main__':
    unittest.main()