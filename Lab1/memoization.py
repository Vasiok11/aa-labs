import time
import sys
import matplotlib.pyplot as plt

sys.setrecursionlimit(100000)


def fibonacci_memoization(n, memo={}):
    if n <= 0:
        return 0
    elif n == 1:
        return 1

    if n in memo:
        return memo[n]

    memo[n] = fibonacci_memoization(n - 1, memo) + fibonacci_memoization(n - 2, memo)

    return memo[n]


def measure_time(func, n):
    start_time = time.time()
    func(n)
    end_time = time.time()
    return end_time - start_time


large_series = [501, 631, 794, 1000, 1259, 1585, 1995, 2512, 3162, 3981, 5012, 6310, 7943, 10000, 12589, 15849]

large_times = []
for n in large_series:
    try:
        time_taken = measure_time(fibonacci_memoization, n)
        large_times.append(time_taken)
    except RecursionError:
        print(f"RecursionError: Skipping n = {n} (too large for recursion)")
        large_times.append(None)

print("Execution Times for Large Series (Memoization Method):")
print("{:<10} {:<15}".format("n", "Time (seconds)"))
for n, t in zip(large_series, large_times):
    if t is not None:
        print("{:<10} {:.6f}".format(n, t))
    else:
        print("{:<10} Skipped (RecursionError)".format(n))

plt.figure(figsize=(8, 6))

valid_n = [n for n, t in zip(large_series, large_times) if t is not None]
valid_times = [t for t in large_times if t is not None]

plt.plot(valid_n, valid_times, marker='o', label='Memoization Method')
plt.xlabel('n')
plt.ylabel('Time (seconds)')
plt.title('Execution Time for Large Series (Memoization Method)')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()