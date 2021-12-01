# Uses python3
import sys

def fibonacci_partial_sum_naive(from_, to):
    _sum = 0

    current = 0
    _next  = 1

    for i in range(to + 1):
        if i >= from_:
            _sum += current

        current, _next = _next, current + _next

    return _sum % 10

def calc_pisano(m):
    previous, current = 0, 1
    for i in range(0, m * m):
        previous, current \
            = current, (previous + current) % m

        # A Pisano Period starts with 01
        if previous == 0 and current == 1:
            return i + 1


def get_fibonacci_huge_fast(n, m):
    k = calc_pisano(m)
    n = n % k
    previous, current = 0, 1
    if n == 0:
        return 0
    elif n == 1:
        return 1
    for i in range(n - 1):
        previous, current \
            = current, previous + current

    return current % m


def fibonacci_partial_sum_smart(m, n):
    x1 = get_fibonacci_huge_fast(m,10)
    if m > 0:
        x2 = get_fibonacci_huge_fast(m-1,10)
        total_m = x1 + x2 - 1
    else:
        total_m = 0
    x1 = get_fibonacci_huge_fast(n,10)
    if n > 0:
        x2 = get_fibonacci_huge_fast(n-1,10)
        total_n = (2 * x1) + x2 - 1
    else:
        total_n = 0
    return (total_n - total_m) % 10

if __name__ == '__main__':
    input = sys.stdin.read();
    from_, to = map(int, input.split())
    print(fibonacci_partial_sum_smart(from_, to))
#    print(fibonacci_partial_sum_naive(from_, to))
