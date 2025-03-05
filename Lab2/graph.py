import tkinter as tk
from tkinter import ttk, messagebox
import random
import time
import threading
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure


# Sorting algorithms implementations (original versions)
def quick_sort(arr):
    def _quick_sort(array, low, high):
        if low < high:
            pi = partition(array, low, high)
            _quick_sort(array, low, pi - 1)
            _quick_sort(array, pi + 1, high)

    def partition(array, low, high):
        pivot = array[high]
        i = low - 1
        for j in range(low, high):
            if array[j] <= pivot:
                i += 1
                array[i], array[j] = array[j], array[i]
        array[i + 1], array[high] = array[high], array[i + 1]
        return i + 1

    _quick_sort(arr, 0, len(arr) - 1)


def heap_sort(arr):
    def heapify(array, n, i):
        largest = i
        left = 2 * i + 1
        right = 2 * i + 2
        if left < n and array[left] > array[largest]:
            largest = left
        if right < n and array[right] > array[largest]:
            largest = right
        if largest != i:
            array[i], array[largest] = array[largest], array[i]
            heapify(array, n, largest)

    n = len(arr)
    for i in range(n // 2 - 1, -1, -1):
        heapify(arr, n, i)
    for i in range(n - 1, 0, -1):
        arr[i], arr[0] = arr[0], arr[i]
        heapify(arr, i, 0)


def merge_sort(arr):
    if len(arr) > 1:
        mid = len(arr) // 2
        L = arr[:mid]
        R = arr[mid:]
        merge_sort(L)
        merge_sort(R)
        i = j = k = 0
        while i < len(L) and j < len(R):
            if L[i] <= R[j]:
                arr[k] = L[i]
                i += 1
            else:
                arr[k] = R[j]
                j += 1
            k += 1
        while i < len(L):
            arr[k] = L[i]
            i += 1
            k += 1
        while j < len(R):
            arr[k] = R[j]
            j += 1
            k += 1


def bitonic_sort(arr):
    def _bitonic_sort(array, low, cnt, up):
        if cnt > 1:
            k = cnt // 2
            _bitonic_sort(array, low, k, True)
            _bitonic_sort(array, low + k, k, False)
            _bitonic_merge(array, low, cnt, up)

    def _bitonic_merge(array, low, cnt, up):
        if cnt > 1:
            k = cnt // 2
            for i in range(low, low + k):
                if (array[i] > array[i + k]) == up:
                    array[i], array[i + k] = array[i + k], array[i]
            _bitonic_merge(array, low, k, up)
            _bitonic_merge(array, low + k, k, up)

    n = len(arr)
    _bitonic_sort(arr, 0, n, True)


# Optimized sorting algorithms
def quick_sort_optimized(arr):
    def _quick_sort(array, low, high):
        while low < high:
            if high - low < 16:  # Use insertion sort for small subarrays
                insertion_sort(array, low, high)
                break
            else:
                pivot = median_of_three(array, low, high)  # Better pivot selection
                pi = partition(array, low, high, pivot)
                if pi - low < high - pi:  # Tail recursion optimization
                    _quick_sort(array, low, pi - 1)
                    low = pi + 1
                else:
                    _quick_sort(array, pi + 1, high)
                    high = pi - 1

    def partition(array, low, high, pivot):
        i = low - 1
        for j in range(low, high):
            if array[j] <= pivot:
                i += 1
                array[i], array[j] = array[j], array[i]
        array[i + 1], array[high] = array[high], array[i + 1]
        return i + 1

    def median_of_three(array, low, high):
        mid = (low + high) // 2
        if array[low] > array[mid]:
            array[low], array[mid] = array[mid], array[low]
        if array[low] > array[high]:
            array[low], array[high] = array[high], array[low]
        if array[mid] > array[high]:
            array[mid], array[high] = array[high], array[mid]
        array[mid], array[high] = array[high], array[mid]  # Move median to high
        return array[high]

    def insertion_sort(array, low, high):
        for i in range(low + 1, high + 1):
            key = array[i]
            j = i - 1
            while j >= low and array[j] > key:
                array[j + 1] = array[j]
                j -= 1
            array[j + 1] = key

    _quick_sort(arr, 0, len(arr) - 1)


def merge_sort_optimized(arr):
    def _merge_sort(array, aux, low, high):
        if high - low < 16:  # Use insertion sort for small subarrays
            insertion_sort(array, low, high)
            return
        mid = (low + high) // 2
        _merge_sort(array, aux, low, mid)
        _merge_sort(array, aux, mid + 1, high)
        if array[mid] <= array[mid + 1]:  # Skip merge if already sorted
            return
        merge(array, aux, low, mid, high)

    def merge(array, aux, low, mid, high):
        for i in range(low, high + 1):
            aux[i] = array[i]
        i, j, k = low, mid + 1, low
        while i <= mid and j <= high:
            if aux[i] <= aux[j]:
                array[k] = aux[i]
                i += 1
            else:
                array[k] = aux[j]
                j += 1
            k += 1
        while i <= mid:
            array[k] = aux[i]
            i += 1
            k += 1

    def insertion_sort(array, low, high):
        for i in range(low + 1, high + 1):
            key = array[i]
            j = i - 1
            while j >= low and array[j] > key:
                array[j + 1] = array[j]
                j -= 1
            array[j + 1] = key

    aux = arr.copy()
    _merge_sort(arr, aux, 0, len(arr) - 1)


def heap_sort_optimized(arr):
    def heapify(array, n, i):
        while True:
            largest = i
            left = 2 * i + 1
            right = 2 * i + 2
            if left < n and array[left] > array[largest]:
                largest = left
            if right < n and array[right] > array[largest]:
                largest = right
            if largest == i:
                break
            array[i], array[largest] = array[largest], array[i]
            i = largest

    n = len(arr)
    # Build heap (bottom-up)
    for i in range(n // 2 - 1, -1, -1):
        heapify(arr, n, i)
    # Extract elements
    for i in range(n - 1, 0, -1):
        arr[i], arr[0] = arr[0], arr[i]
        heapify(arr, i, 0)


def bitonic_sort_optimized(arr):
    def _bitonic_merge(array, low, cnt, up):
        if cnt > 1:
            k = cnt // 2
            for i in range(low, low + k):
                if (array[i] > array[i + k]) == up:
                    array[i], array[i + k] = array[i + k], array[i]
            _bitonic_merge(array, low, k, up)
            _bitonic_merge(array, low + k, k, up)

    n = len(arr)
    # Ensure n is a power of 2
    if not (n & (n - 1)) == 0:
        next_power = 1 << (n - 1).bit_length()
        arr.extend([float('inf')] * (next_power - n))  # Pad with infinity
        n = next_power

    # Iterative bitonic sort
    size = 2
    while size <= n:
        for low in range(0, n, size):
            mid = low + size // 2
            _bitonic_merge(arr, low, size, True)
        size *= 2

    # Remove padding
    while len(arr) > n:
        arr.pop()


class SortingVisualizer:
    def __init__(self, root):
        self.root = root
        self.root.title("Sorting Algorithm Performance")
        self.root.geometry("1200x900")
        self.root.configure(bg="#f0f0f0")

        self.setup_controls()
        self.setup_plot()

        self.test_thread = None
        self.running = False
        self.optimized_mode = False  # Toggle for optimized algorithms

    def setup_controls(self):
        control_frame = ttk.Frame(self.root, padding="10")
        control_frame.pack(pady=10, fill=tk.X)

        # Min array size
        ttk.Label(control_frame, text="Min Size:").grid(row=0, column=0, padx=5)
        self.min_entry = ttk.Entry(control_frame, width=10)
        self.min_entry.grid(row=0, column=1, padx=5)

        # Max array size
        ttk.Label(control_frame, text="Max Size:").grid(row=0, column=2, padx=5)
        self.max_entry = ttk.Entry(control_frame, width=10)
        self.max_entry.grid(row=0, column=3, padx=5)

        # Number of arrays
        ttk.Label(control_frame, text="Number of Arrays:").grid(row=0, column=4, padx=5)
        self.num_arrays_entry = ttk.Entry(control_frame, width=10)
        self.num_arrays_entry.grid(row=0, column=5, padx=5)

        # Min value input
        ttk.Label(control_frame, text="Min Value:").grid(row=1, column=0, padx=5)
        self.min_val_entry = ttk.Entry(control_frame, width=10)
        self.min_val_entry.grid(row=1, column=1, padx=5)
        self.min_val_entry.insert(0, "-1000")

        # Max value input
        ttk.Label(control_frame, text="Max Value:").grid(row=1, column=2, padx=5)
        self.max_val_entry = ttk.Entry(control_frame, width=10)
        self.max_val_entry.grid(row=1, column=3, padx=5)
        self.max_val_entry.insert(0, "1000")

        # Allow negative numbers checkbox
        self.neg_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(control_frame, text="Allow Negative Numbers", variable=self.neg_var).grid(row=1, column=4, padx=5)

        # Algorithms listbox
        ttk.Label(control_frame, text="Algorithms:").grid(row=1, column=5, padx=5)
        self.algorithms_listbox = tk.Listbox(control_frame, selectmode=tk.MULTIPLE, height=4, width=15)
        self.algorithms_listbox.grid(row=1, column=6, padx=5)
        for algo in ["Quick Sort", "Merge Sort", "Heap Sort", "Bitonic Sort"]:
            self.algorithms_listbox.insert(tk.END, algo)

        # Optimized Algorithm Mode toggle
        self.optimized_var = tk.BooleanVar()
        ttk.Checkbutton(control_frame, text="Optimized Algorithm Mode", variable=self.optimized_var).grid(row=1, column=7, padx=5)

        # Run button
        self.run_button = ttk.Button(control_frame, text="Run Tests", command=self.start_tests)
        self.run_button.grid(row=1, column=8, padx=5)

        # Reset button
        ttk.Button(control_frame, text="Reset", command=self.reset).grid(row=1, column=9, padx=5)

        # Bitonic Array Mode toggle
        self.bitonic_mode_var = tk.BooleanVar()
        ttk.Checkbutton(control_frame, text="Bitonic Array Mode", variable=self.bitonic_mode_var).grid(row=1, column=10,
                                                                                                       padx=5)

    def setup_plot(self):
        self.figure = Figure(figsize=(10, 6), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.root)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Add toolbar
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.root)
        self.toolbar.update()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    def is_power_of_two(self, n):
        return (n & (n - 1)) == 0 and n != 0

    def start_tests(self):
        if self.running:
            return
        try:
            min_size = int(self.min_entry.get())
            max_size = int(self.max_entry.get())
            num_points = int(self.num_arrays_entry.get())
            min_val = int(self.min_val_entry.get())
            max_val = int(self.max_val_entry.get())
            allow_neg = self.neg_var.get()
            selected = self.algorithms_listbox.curselection()
            algorithms = [self.algorithms_listbox.get(i) for i in selected]
            self.optimized_mode = self.optimized_var.get()
            bitonic_mode = self.bitonic_mode_var.get()  # Check Bitonic Array Mode
        except ValueError:
            messagebox.showerror("Error", "Invalid input values")
            return

        if not algorithms:
            messagebox.showerror("Error", "Select at least one algorithm")
            return

        if min_size <= 0 or max_size <= min_size or num_points <= 0:
            messagebox.showerror("Error", "Invalid size range or number of points")
            return

        if not allow_neg and min_val < 0:
            messagebox.showerror("Error", "Min value cannot be negative when negative numbers are disabled.")
            return

        # Generate test sizes
        if bitonic_mode:
            sizes = self.generate_power_of_two_sizes(min_size, max_size, num_points)
        else:
            if num_points == 1:
                sizes = [min_size]
            else:
                step = (max_size - min_size) / (num_points - 1)
                sizes = [int(min_size + i * step) for i in range(num_points)]

        self.running = True
        self.run_button.config(state=tk.DISABLED)

        # Start test thread
        self.test_thread = threading.Thread(
            target=self.run_tests,
            args=(sizes, algorithms, min_val, max_val, allow_neg),
            daemon=True
        )
        self.test_thread.start()

        # Start monitoring the thread
        self.monitor_thread()

    def monitor_thread(self):
        if self.test_thread.is_alive():
            self.root.after(100, self.monitor_thread)
        else:
            self.running = False
            self.run_button.config(state=tk.NORMAL)

    def generate_power_of_two_sizes(self, min_size, max_size, num_points):
        """
        Generates a list of array sizes that are powers of two within the specified range.
        """
        sizes = []
        current_size = 1
        while current_size <= max_size:
            if current_size >= min_size:
                sizes.append(current_size)
            current_size *= 2
            if len(sizes) >= num_points:
                break
        return sizes

    def run_tests(self, sizes, algorithms, min_val, max_val, allow_neg):
        data = {algo: {'sizes': [], 'times': []} for algo in algorithms}
        if self.optimized_mode:
            for algo in algorithms:
                data[f"Optimized {algo}"] = {'sizes': [], 'times': []}

        for size in sizes:
            arr = [random.randint(min_val, max_val) for _ in range(size)]
            if not allow_neg:
                arr = [abs(num) for num in arr]

            for algo in algorithms:
                if algo == "Bitonic Sort" and not self.is_power_of_two(size):
                    continue  # Skip invalid sizes for Bitonic Sort

                # Test original algorithm
                arr_copy = arr.copy()
                start_time = time.perf_counter()
                if algo == "Quick Sort":
                    quick_sort(arr_copy)
                elif algo == "Merge Sort":
                    merge_sort(arr_copy)
                elif algo == "Heap Sort":
                    heap_sort(arr_copy)
                elif algo == "Bitonic Sort":
                    bitonic_sort(arr_copy)
                elapsed = time.perf_counter() - start_time
                data[algo]['sizes'].append(size)
                data[algo]['times'].append(elapsed)

                # Test optimized algorithm if enabled
                if self.optimized_mode:
                    arr_copy = arr.copy()
                    start_time = time.perf_counter()
                    if algo == "Quick Sort":
                        quick_sort_optimized(arr_copy)
                    elif algo == "Merge Sort":
                        merge_sort_optimized(arr_copy)
                    elif algo == "Heap Sort":
                        heap_sort_optimized(arr_copy)
                    elif algo == "Bitonic Sort":
                        bitonic_sort_optimized(arr_copy)
                    elapsed = time.perf_counter() - start_time
                    data[f"Optimized {algo}"]['sizes'].append(size)
                    data[f"Optimized {algo}"]['times'].append(elapsed)

        # Update plot in main thread
        self.root.after(0, self.update_plot, data)

    def update_plot(self, data):
        self.figure.clf()
        ax = self.figure.add_subplot(111)

        for algo in data:
            if not data[algo]['sizes']:
                continue
            ax.plot(
                data[algo]['sizes'],
                data[algo]['times'],
                marker='o',
                linestyle='-',
                label=algo
            )

        ax.set_xlabel("Array Size")
        ax.set_ylabel("Time (seconds)")
        ax.set_title("Sorting Algorithm Performance Comparison")
        ax.legend()
        ax.grid(True)
        self.canvas.draw()

    def reset(self):
        self.figure.clf()
        self.canvas.draw()
        self.min_entry.delete(0, tk.END)
        self.max_entry.delete(0, tk.END)
        self.num_arrays_entry.delete(0, tk.END)
        self.min_val_entry.delete(0, tk.END)
        self.max_val_entry.delete(0, tk.END)
        self.min_val_entry.insert(0, "-1000")
        self.max_val_entry.insert(0, "1000")
        self.algorithms_listbox.selection_clear(0, tk.END)
        self.optimized_var.set(False)


if __name__ == "__main__":
    root = tk.Tk()
    app = SortingVisualizer(root)
    root.mainloop()