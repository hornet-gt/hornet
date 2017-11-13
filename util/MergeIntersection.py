#!/usr/bin/python
import unittest 

# |len(u)| <= |len(v)| input assumption
def countIntersection(u, v, threads_per_intersect):

    total_work = len(u)+len(v)
    work_per_thread = max(total_work/threads_per_intersect, 1)
    diag_ids = []
    id_assigned = 0
    while (id_assigned+work_per_thread) < total_work-1:
        id_assigned += work_per_thread
        diag_ids.append(id_assigned)

    # print('diag_ids', diag_ids)
    triangle_count = 0
    pathPoints = []
    ui_start, vi_start, ui_end, vi_end = -1, -1, -1, -1
    for diag_id in diag_ids:
        if diag_id < len(u):
            # print diag_id, '< len u'
            ui_start, ui_end = diag_id-1, 0
            vi_start, vi_end = 0, diag_id-1
        elif diag_id < len(v):
            # print diag_id, '< len v'
            ui_start, ui_end = len(u)-1, 0
            vi_start, vi_end = diag_id-len(u), diag_id-1
        else:
            # print diag_id, '>= both'
            ui_start, ui_end = len(u) - 1, diag_id - len(v)
            vi_start, vi_end = diag_id-len(u), len(v)-1
        # print("bSearch: (v_start: {}, u_start: {}) -> (v_end: {}, u_end: {})".format(vi_start, ui_start, vi_end, ui_end))
        vi_pathPos, ui_pathPos = bSearch(u, v, vi_start, ui_start, vi_end, ui_end)
        # print("bSearch found: ({}, {})".format(vi_pathPos, ui_pathPos))
        pathPoints.append((vi_pathPos, ui_pathPos))

    pathPoints.append((len(v)-1, len(u)-1)) # endpoint diagonal
    # handle (0,0) case separately
    triangle_count += intersect(u, v, 0, 0, pathPoints[0][0], pathPoints[0][1])
    for i in range(len(pathPoints)-1): 
        vi_curr, ui_curr = pathPoints[i]
        vi_next, ui_next = pathPoints[i+1]
        vi_inBounds = (vi_curr < len(v)-1)
        ui_inBounds = (ui_curr < len(u)-1)

        vi_begin, ui_begin = vi_curr, ui_curr
        # TODO: refactor to minimize thread divergence?
        if vi_inBounds and ui_inBounds:
            comp = (u[ui_curr+1] >= v[vi_curr+1]) # >= (~)
            vi_begin += int(comp)
            ui_begin += int(not comp)
        else:
            vi_begin += int(vi_inBounds)
            ui_begin += int(ui_inBounds)

        triangle_count += intersect(u, v, vi_begin, ui_begin, vi_next, ui_next)
    return triangle_count


def bSearch(u, v, low_vi, low_ui, high_vi, high_ui):

    while True:
        mid_ui = int((low_ui + high_ui) / 2) # round down
        mid_vi = int((low_vi + high_vi + 1) / 2) # round up

        # comparisons designed to handle equality
        comp1 = (u[mid_ui] < v[mid_vi])
        # 1-element diagonal case
        if (low_ui == high_ui and low_vi == high_vi):
            return ((mid_vi, mid_ui))

        if not comp1: # comp1 is +, comp2 is +
            low_ui = mid_ui
            low_vi = mid_vi
            continue

        # 2-element diagonal base case
        comp2 = (u[mid_ui+1] >= v[mid_vi-1])
        if comp1 and not comp2: # comp1 is -, comp2 is -
            high_ui = mid_ui+1
            high_vi = mid_vi-1
        elif comp1 and comp2: # comp1 is -, comp2 is +; DONE
            comp3 = (u[mid_ui+1] < v[mid_vi])
            return (mid_vi - int(comp3), mid_ui + int(comp3))


def intersect(u, v, vi_start, ui_start, vi_end, ui_end):
    count = 0
    # print("({}, {}) -> ({}, {})".format(vi_start, ui_start, vi_end, ui_end))
    while (vi_start <= vi_end and ui_start <= ui_end):
        # print("({}, {}), comparing ({}, {})".format(vi_start, ui_start, v[vi_start], u[ui_start]))
        comp_equals = (u[ui_start] == v[vi_start])
        if comp_equals:
            count += 1
        comp1 = u[ui_start] >= v[vi_start]
        comp2 = u[ui_start] <= v[vi_start]
        ui_bound = (ui_start == ui_end)
        vi_bound = (vi_start == vi_end)
        if (ui_bound and comp2) or (vi_bound and comp1): # early termination (optimization)
            break
        if (comp1 and not vi_bound) or ui_bound:
            vi_start += 1
        if (comp2 and not ui_bound) or vi_bound:
            ui_start += 1
        # print("after: (vi_start: {}, ui_start {})".format(vi_start, ui_start))
    return count
