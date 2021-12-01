# Uses python3
import sys

def fibonacci_sum_naive(n):
    if n <= 1:
        return n

    previous = 0
    current  = 1
    _sum      = 1

    for _ in range(n - 1):
        previous, current = current, previous + current
        _sum += current

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


def fibonacci_sum_smart(n):
    x1 = get_fibonacci_huge_fast(n,10)
    if n > 0:
        x2 = get_fibonacci_huge_fast(n-1,10)
    else:
        return 0
    total = (2 * x1) + x2 - 1
    return total % 10


if __name__ == '__main__':
    input = sys.stdin.read()
    n = int(input)
    print(fibonacci_sum_smart(n))
#    print(fibonacci_sum_naive(n))
