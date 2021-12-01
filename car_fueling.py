# python3
import sys


def compute_min_refills(distance, tank, stops):
    n = len(stops)
    next_stop = 0
    total_stops = 0
    position = 0
    safe_max = 0
    #traverse to next stop
    while next_stop < n and position < distance:
        assert next_stop < n
        assert position < distance
        if stops[next_stop] - position <= tank:
            safe_max = stops[next_stop]
            next_stop = next_stop + 1
            # print(f"safe station at {safe_max}")
        elif safe_max + tank >= stops[next_stop]:
            position = stops[next_stop - 1]
            total_stops = total_stops + 1
            safe_max = stops[next_stop]
            next_stop = next_stop + 1
            # print(f"stopping at {position}")
        else:
            # print("not gonna make it")
            #impossible
            break
    # we will make it!
    if position + tank >= distance:
        return total_stops
    elif safe_max + tank >= distance:
        # print(f"final stop at {safe_max}")
        total_stops = total_stops + 1
        return total_stops

    # didn't make it
    return -1

if __name__ == '__main__':
    d, m, _, *stops = map(int, sys.stdin.read().split())
    print(compute_min_refills(d, m, stops))
