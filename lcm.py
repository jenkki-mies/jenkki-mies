# Uses python3
import sys

def lcm_naive(a, b):
    for l in range(1, a*b + 1):
        if l % a == 0 and l % b == 0:
            return l

    return a*b

def factors(u):
    factorlist = []
    upperlimit = u
    for i in range(2,upperlimit):
        if u % i == 0:
            cnt = 1
            while (u% (i**(cnt+1)) == 0):
                cnt = cnt + 1
            if upperlimit % (i**cnt) == 0:
                upperlimit = upperlimit / (i**cnt)
                for j in range(cnt):
                    factorlist.append(i)
                # print(f"new upperlimit:{upperlimit}")
    return factorlist

def lcm_fast(u, v):
    if u > v:
        a=v
        b=u
    else:
        a=u
        b=v
    fact_a = factors(a)
    if fact_a == []:
        fact_a.append(a)
    b_common = 1
    b_div = b
    for f in fact_a:
        if b_div % f == 0:
            b_div = int(b_div/f)
            b_common = f*b_common
    # print(f"fact_a={fact_a}, a,b={a},{b}, b_div={b_div}, b_common={b_common}")
    return int(a*b/b_common)

if __name__ == '__main__':
    input = sys.stdin.read()
    a, b = map(int, input.split())
    print(lcm_fast(a, b))
