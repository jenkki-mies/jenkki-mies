# Uses python3
import sys

def optimal_summands(n):
    summands = []
    rem = []
    if n == 2 or n == 1:
        summands.append(n)
        return summands
    #write your code here
    for i in range(n):
        if i * (i+1) >= 2*n:
            break
    summands.append(i)
    if n-i > 1:
        rem = optimal_summands(n-i)
    else:
        rem.append(1)
    for x in rem:
        summands.append(x)
    #print(f"summands({n})=",summands)
    return sorted(summands)

if __name__ == '__main__':
    input = sys.stdin.read()
    n = int(input)
    summands = optimal_summands(n)
    print(len(summands))
    for x in summands:
        print(x, end=' ')
