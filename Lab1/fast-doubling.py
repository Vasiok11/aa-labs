import time
import matplotlib.pyplot as plt


def fib_fast_doubling(n):
    def fib(n):
        if n == 0:
            return (0, 1)
        else:
            a, b = fib(n // 2)
            c = a * ((b << 1) - a)
            d = a * a + b * b
            return (d, c + d) if n % 2 else (c, d)

    return fib(n)[0]


def measure_time(func, n):
    start_time = time.time()
    func(n)
    end_time = time.time()
    return end_time - start_time


large_series = [501, 631, 794, 1000, 1259, 1585, 1995, 2512, 3162, 3981, 5012, 6310, 7943, 10000, 12589, 15849]

large_times = [measure_time(fib_fast_doubling, n) for n in large_series]

print("{:<10} {:<15}".format("n", "Time (seconds)"))
for n, t in zip(large_series, large_times):
    print("{:<10} {:.6f}".format(n, t))

plt.figure(figsize=(8, 6))

plt.plot(large_series, large_times, marker='o', label='Fast Doubling Method')
plt.xlabel('n')
plt.ylabel('Time (seconds)')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()