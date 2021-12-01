# Uses python3
import sys
from collections import namedtuple

Segment = namedtuple('Segment', 'start end')

def seg_coverage(points, val1, val2):
    for p in points:
        if val1 <= p and val2 >= p:
            return True
    return False

def pair_coverage_check(points, pair):
     for p in points:
         if pair[0] <= p and pair[1] >= p:
             return True
     return False

def optimal_points(segments):
    startpts = []
    endpts = []
    points1 = []
    points2 = []
    #write your code here
    for s in segments:
        startpts.append(s.start)
        endpts.append(s.end)
    pairs_bystart = sorted(zip(startpts, endpts))
    pairs_byend = sorted(pairs_bystart, key=lambda seg: seg[1], reverse=False)
    print("pairs by start:", pairs_bystart)
    print("pairs by end:", pairs_byend)
    for p in pairs_bystart:
        if pair_coverage_check(points1, p):
            continue
        else:
            points1.append(p[0])
    for p in pairs_byend:
        if pair_coverage_check(points2, p):
            continue
        else:
            points2.append(p[1])
    if len(points1) > len(points2):
        print("difference")
        return points2
    else:
        return points1

if __name__ == '__main__':
    input = sys.stdin.read()
    n, *data = map(int, input.split())
    segments = list(map(lambda x: Segment(x[0], x[1]), zip(data[::2], data[1::2])))
    points = optimal_points(segments)
    print(len(points))
    print(*points)
