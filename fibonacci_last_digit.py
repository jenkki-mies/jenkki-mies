# Uses python3
import sys
# import time

# def get_fibonacci_last_digit_naive(n):
#     if n <= 1:
#         return n
#
#     previous = 0
#     current  = 1
#
#     for _ in range(n - 1):
#         previous, current = current, previous + current
#
#     return current % 10

def get_fibonacci_last_digit_fast(n):
    if n <= 1:
        return n

    previous = 0
    current  = 1

    for _ in range(n - 1):
        previous, current = current, (previous + current) % 10

    return current

if __name__ == '__main__':
    # n = int(input())
    # print("got the integer", n)
    # start = time.perf_counter()
    # slow = get_fibonacci_last_digit_naive(n)
    # stop = time.perf_counter()
    # print(f"slow implementation fiblastdigit{n}: {slow} took {stop-start} seconds")
    # start = time.perf_counter()
    # fast = get_fibonacci_last_digit_fast(n)
    # stop = time.perf_counter()
    # print(f"fast implementation fiblastdigit{n}: {fast} took {stop-start} seconds")
    # print("testing with sys.stdin.read() input")
    input: str = sys.stdin.read()
    # print("got the following input:", input)
    n = int(input)
    fast = get_fibonacci_last_digit_fast(n)
    print(fast)
