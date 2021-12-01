#Uses python3

import sys

import numpy as np


def DecimalIntegerGrThOrEq(a, b):
    if b == None:
        return True
    A = int(a)
    B = int(b)
    #print(f"Testing GrE {A} vs {B}")
    digits_in_a = np.floor(np.log10(A))
    digits_in_b = np.floor(np.log10(B))
    if digits_in_a == digits_in_b:
        #print(f"comparator: {A} vs {B}")
        flag = A >= B
    else:
        comp_a = int(A / (10 ** digits_in_a))
        comp_b = int(B / (10 ** digits_in_b))
        #print(f"comparator digits: {comp_a}/{digits_in_a}  vs {comp_b}/{digits_in_b}")
        if digits_in_a < digits_in_b:
            flag = comp_a >= comp_b
        else:
            flag = comp_b < comp_a
    #print(f"comparator returning {flag}")
    return flag


def largest_number(a):
    res = ""
    while len(a) > 0:
        max = None
        for digit in a:
            if DecimalIntegerGrThOrEq(digit, max):
                max = digit
        res += max
        a.remove(max)
    return res

if __name__ == '__main__':
    input = sys.stdin.read()
    data = input.split()
    a = data[1:]
    print(largest_number(a))
    
