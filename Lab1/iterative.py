import time
import matplotlib.pyplot as plt


def fibonacci_iterative(n):
    if n <= 0:
        return 0
    elif n == 1:
        return 1

    current_term = 1
    previous_term = 1
    second_previous_term = 0

    for _ in range(2, n):
        second_previous_term = previous_term
        previous_term = current_term
        current_term = second_previous_term + previous_term

    return current_term


def measure_time(func, n):
    start_time = time.time()
    func(n)
    end_time = time.time()
    return end_time - start_time


series = [501, 631, 794, 1000, 1259, 1585, 1995, 2512, 3162, 3981, 5012, 6310, 7943, 10000, 12589, 15849]

times = [measure_time(fibonacci_iterative, n) for n in series]

print("{:<10} {:<15}".format("n", "Time (seconds)"))
for n, t in zip(series, times):
    print("{:<10} {:.6f}".format(n, t))

plt.figure(figsize=(8, 6))

plt.plot(series, times, marker='o', label='Iterative Method')
plt.xlabel('n')
plt.ylabel('Time (seconds)')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()