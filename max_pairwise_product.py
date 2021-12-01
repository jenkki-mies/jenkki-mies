import numpy as np
from heapq import nlargest

def orig_max_pairwise_product(numbers):
    n = len(numbers)
    max_product = 0
    for first in range(n):
        for second in range(first + 1, n):
            max_product = max(max_product,
                numbers[first] * numbers[second])

    return max_product

def max_pairwise_product(numbers):
    max1 = max(numbers)
    numbers.remove(max1)
    max2 = max(numbers)
    return max1 * max2

def max_pairwise_product_needswork(numbers):
    list = nlargest(2,numbers)

##if __name__ == '__main__':
##    MAX = 2e5+1
##    print("testing heapq")
##    test_numbers = np.arange(MAX)
##    teval = max_pairwise_product(test_numbers.tolist())
##    print(f'the max_pairwise_product({MAX})={teval}')
    
if __name__ == '__main__':
    MAX = 2e5+1
    input_n = int(input())
    input_numbers = [int(x) for x in input().split()]
    eval = max_pairwise_product(input_numbers)
    print(eval)
