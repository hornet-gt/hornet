#!/usr/bin/python

# |len(u)| <= |len(v)| input assumption
def countIntersection(u, v, threads_per_intersect):
    total_work = len(u)+len(v)
    work_per_thread = total_work/threads_per_intersect
    diag_ids = []

    for i in range(threads_per_intersect):
        diag_ids.append(i*work_per_thread)

    print('diag_ids', diag_ids)

    triangle_count = 0
    pathPoints = []
    ui_start, vi_start, ui_end, vi_end = -1, -1, -1, -1

    pathPoints.append((0,0)) # startpoint diagonal
    for diag_id in diag_ids[1:]:
        if diag_id < len(u):
            print diag_id, '< len u'
            ui_start, ui_end = diag_id-1, 0
            vi_start, vi_end = 0, diag_id-1
        elif diag_id < len(v):
            print diag_id, '< len v'
            ui_start, ui_end = len(u)-1, 0
            vi_start, vi_end = diag_id - len(u), diag_id-1
        else:
            print diag_id, '>= both'
            #ui_start, ui_end = len(u)-1, diag_id - len(v)
            #vi_start, vi_end = diag_id - len(v), len(v)-1
            ui_start, ui_end = len(u) - 1, diag_id - len(v)
            vi_start, vi_end = diag_id-len(u), len(v)-1
        print("bSearch: (v_start: {}, u_start: {}) -> (v_end: {}, u_end: {})".format(vi_start, ui_start, vi_end, ui_end))
        vi_pathPos, ui_pathPos = bSearch(u, v, vi_start, ui_start, vi_end, ui_end)
        pathPoints.append((vi_pathPos, ui_pathPos))

    pathPoints.append((len(v)-1, len(u)-1)) # endpoint diagonal
    for i in range(len(pathPoints)-1): 
        pt_curr = pathPoints[i]
        pt_next = pathPoints[i+1]
        vi_offset = int((pt_next[0] - pt_curr[0]) > 0)
        ui_offset = int((pt_next[1] - pt_curr[1]) > 0)
        triangle_count += intersectCount(u, v, pt_curr[0], pt_curr[1], pt_next[0] - vi_offset, pt_next[1] - ui_offset)
    return triangle_count


def bSearch(u, v, low_vi, low_ui, high_vi, high_ui):

    while True:
        mid_ui = int((low_ui + high_ui) / 2) # round down
        mid_vi = int((low_vi + high_vi + 1) / 2) # round up

        # comparisons designed to handle equality
        comp1 = (u[mid_ui] < v[mid_vi])
        # 1-element diagonal case
        if (low_ui == high_ui and low_vi == high_vi): 
            if not comp1: 
                return (min(mid_vi+1, len(v)-1), mid_ui)
            else:
                return (mid_vi, min(mid_ui+1, len(u)-1))


        # 2-element diagonal base case
        comp2 = (u[mid_ui+1] >= v[mid_vi-1])
        if comp1 and comp2: # comp1 is -, comp2 is +; DONE
            return (mid_vi, mid_ui+1)
        elif not comp1: # comp1 is +, comp2 is +
            if (high_ui == mid_ui and high_vi == mid_vi): # boundary condition case
                return (min(mid_vi+1, len(v)-1), mid_ui)
            else: # continue bSearch
                low_ui = mid_ui-1
                low_vi = mid_vi+1
        elif comp1 and not comp2: # comp1 is -, comp2 is -
            if (low_ui == mid_ui and low_vi == mid_vi):
                return (mid_vi, min(mid_ui+1, len(u)-1))
            else:
                high_ui = mid_ui+1
                high_vi = mid_vi-1


# # API update? Providing subarray endpoints may require synchronization
# def intersectCount(u, v, vi_pathPos, ui_pathPos, nComps):
#   count = 0
#   while (nComps > 0):
#       comp = (u[ui_pathPos] == v[vi_pathPos])
#       if comp:
#           count += 1
#       if (ui_pathPos == len(u) or u[ui_pathPos] >= v[vi_pathPos]):
#           vi_pathPos += 1
#       else: # check logic
#           ui_pathPos += 1
#       nComps -= 1
#   return count

def intersectCount(u, v, vi_start, ui_start, vi_end, ui_end):
    count = 0
    print("({}, {}) -> ({}, {})".format(vi_start, ui_start, vi_end, ui_end))
    while (vi_start <= vi_end and ui_start <= vi_end):
        print("({}, {})".format(vi_start, ui_start))
        comp = (u[ui_start] == v[vi_start])
        if comp:
            count += 1
        if ui_start == len(u) or u[ui_start] >= v[vi_start]:
            vi_start += 1
        else:
            ui_start += 1
    return count

def testBinarySearch():
    u, v = [2, 4, 8], [1, 2, 5, 7, 8] 
    threads_per_intersect = 4
    print(bSearch(u, v, 0, 1, 1, 0))
    print(bSearch(u, v, 1, 2, 3, 0))
    print(bSearch(u, v, 3, 2, 4, 1))
    print("\n")

    u, v = [6, 7, 8], [1, 2, 3, 4, 5] 
    threads_per_intersect = 4
    print(bSearch(u, v, 0, 1, 1, 0))
    print(bSearch(u, v, 1, 2, 3, 0))
    print(bSearch(u, v, 3, 2, 4, 1))
    print("\n")

    u, v = [1, 2, 3], [4, 5, 6, 7, 8] 
    threads_per_intersect = 4
    print(bSearch(u, v, 0, 1, 1, 0))
    print(bSearch(u, v, 1, 2, 3, 0))
    print(bSearch(u, v, 3, 2, 4, 1))
    print("\n")

## testing binary search
# testBinarySearch()

# testing intersection
# NOTE: works if we give accurate nComp; possible to calculate if we have all the path points
u, v = [2, 4, 8], [1, 2, 5, 7, 8] 
v0, u0 = (0,0)
v1, u1 = bSearch(u, v, 0, 1, 1, 0)
v2, u2 = bSearch(u, v, 1, 2, 3, 0)
v3, u3 = bSearch(u, v, 3, 2, 4, 1)
v4, u4 = (4, 2)
# print(intersectCount(u, v, v0, u0, 2))
# print(intersectCount(u, v, v1, u1, 2))
# print(intersectCount(u, v, v2, u2, 2))
# print(intersectCount(u, v, v3, u3, 1))
print(intersectCount(u, v, v0, u0, v1-1, u1))
print(intersectCount(u, v, v1, u1, v2, u2-1))
print(intersectCount(u, v, v2, u2, v3-1, u3))
print(intersectCount(u, v, v3, u3, v4, u4))

print("Final count: {}".format(countIntersection(u, v, 4)))

