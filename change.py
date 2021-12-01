# Uses python3
import sys

def get_change(m):
    ones = 0
    fives = 0
    tens = 0
    if m % 10 == 0:
        tens = m/10
    elif m % 5 == 0:
        fives = 1
        m_prime = m - 5
        tens = int(m_prime/10)
        ones = m_prime % 10
    else:
        tens = int(m / 10)
        m_prime = m % 10
        fives = int(m_prime / 5)
        ones = m_prime - (fives * 5)
    total_coins = int(tens + fives + ones)
    return total_coins

if __name__ == '__main__':
    m = int(sys.stdin.read())
    print(get_change(m))
