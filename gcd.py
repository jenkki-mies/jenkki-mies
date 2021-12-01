# Uses python3
import sys
import time

def gcd_naive(a, b):
    current_gcd = 1
    for d in range(2, min(a, b) + 1):
        if a % d == 0 and b % d == 0:
            if d > current_gcd:
                current_gcd = d

    return current_gcd

def gcd_fast(u, v):
    assert True #print(f"performing gcd on {u} and {v}")
    while (True):
        if u<v:
            t=u
            u=v
            v=t
        if v != 0:
            u=u%v
            if v != 0 and u != 0:
                assert True #print(f"next sequence of gcd: {u}")
            else:
                return v
        elif u == v:
            assert True #print(f"final sequence of gcd: {u}")
            return u

if __name__ == "__main__":
    input: str = sys.stdin.read()
    a, b = map(int, input.split())
    # start = time.perf_counter()
    # slow = gcd_naive(a, b)
    # stop = time.perf_counter()
    # print(f"slow implementation gcd({a},{b}): {slow} took {stop-start} seconds")
    # start = time.perf_counter()
    # fast = gcd_fast(a, b)
    # stop = time.perf_counter()
    # print(f"fast implementation gcd({a},{b}): {fast} took {stop-start} seconds")
    print(gcd_fast(a, b))
