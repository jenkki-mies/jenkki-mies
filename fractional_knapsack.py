# Uses python3
import sys

import numpy as np


def get_optimal_value_naive(capacity, weights, values):
    value = 0.
    # write your code here
    return value

def get_optimal_value(capacity, weights, values):
    items = []
    value = 0
    left_capacity = capacity
    n = len(weights)
    val_array = np.array(values)
    weight_array = np.array(weights)
    val_density = val_array / weight_array
    val_density_list = val_density.tolist()
    pairs = sorted(zip(val_density_list, range(n)),reverse=True)
    r = [x[1] for x in pairs]
    # print(f"density list order 0 to {N-1}: : {r}")
    # print(f"original list: {val_density_list}")
    for i in r:
         w = weight_array[i]
         v = val_array[i]
         if (left_capacity - w >= 0):
             #adding the items
             value = value + v
             left_capacity = left_capacity - w
             items.append((v, w))
             if left_capacity == 0:
                 break
         else:
            fractional_val = left_capacity * val_density[i]
            value = value + fractional_val
            items.append((fractional_val, left_capacity))
            left_capacity = 0
            break
    # print("Knapsack contents:", items)
    # print("Total weight: ", capacity - left_capacity)
    # print("Total value: ", value)

    return value

if __name__ == "__main__":
    ##sys.stdin.readline()[:-1]
    data = list(map(int, sys.stdin.read().split()))
    n, capacity = data[0:2]
    values = data[2:(2 * n + 2):2]
    weights = data[3:(2 * n + 2):2]
    opt_value = get_optimal_value(capacity, weights, values)
    print("{:.4f}".format(opt_value))
