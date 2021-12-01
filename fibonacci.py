# Uses python3

def calc_fib(n):
    if n == 0:
        sum = 0
    else:
        x0 = 0
        x1 = 1
        sum = 1
        for i in range(n-1):
            sum = x0 + x1
            x0 = x1
            x1 = sum

    return sum

##official main:
max = 45
min = 0
n = int(input())
if n <= max and n >= min:
    print(calc_fib(n))
else:
    print(f"{n} was out of range")
