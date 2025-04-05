import math

def binomial_coefficient(n, k):
    return math.comb(n, k)

def total_sequences(n, lower_bound, upper_bound):
    total = 0
    for k in range(lower_bound, upper_bound + 1):
        total += binomial_coefficient(n, k)
    return total

n = 200
lower_bound = int((0.67 - 0.07) * n)
upper_bound = int((0.67 + 0.07) * n)

total = total_sequences(n, lower_bound, upper_bound)
print(total)
