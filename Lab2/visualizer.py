import tkinter as tk
from tkinter import ttk, messagebox
import random
import time
from tkinter import font as tkFont


# Sorting algorithms implementations (original versions)
def quick_sort_gen(array):
    def _quick_sort(arr, low, high):
        if low < high:
            pivot = arr[high]
            i = low - 1
            for j in range(low, high):
                if arr[j] <= pivot:
                    i += 1
                    arr[i], arr[j] = arr[j], arr[i]
                    yield arr, i, j
            arr[i + 1], arr[high] = arr[high], arr[i + 1]
            yield arr, i + 1, high
            yield from _quick_sort(arr, low, i)
            yield from _quick_sort(arr, i + 2, high)

    yield from _quick_sort(array, 0, len(array) - 1)


def heap_sort_gen(array):
    def heapify(arr, n, i):
        largest = i
        l = 2 * i + 1
        r = 2 * i + 2

        if l < n and arr[l] > arr[largest]:
            largest = l
        if r < n and arr[r] > arr[largest]:
            largest = r
        if largest != i:
            arr[i], arr[largest] = arr[largest], arr[i]
            yield arr, i, largest
            yield from heapify(arr, n, largest)

    n = len(array)
    for i in range(n // 2 - 1, -1, -1):
        yield from heapify(array, n, i)

    for i in range(n - 1, 0, -1):
        array[0], array[i] = array[i], array[0]
        yield array, 0, i
        yield from heapify(array, i, 0)


def merge_sort_gen(array):
    def _merge_sort(arr, l, r):
        if l < r:
            m = (l + r) // 2
            yield from _merge_sort(arr, l, m)
            yield from _merge_sort(arr, m + 1, r)
            yield from merge(arr, l, m, r)

    def merge(arr, l, m, r):
        temp = arr[l:r + 1]
        i, j, k = 0, m - l + 1, l
        max_i = m - l
        max_j = r - l

        while i <= max_i and j <= max_j:
            if temp[i] <= temp[j]:
                arr[k] = temp[i]
                i += 1
            else:
                arr[k] = temp[j]
                j += 1
            k += 1
            yield arr, k - 1, j + l - 1

        while i <= max_i:
            arr[k] = temp[i]
            i += 1
            k += 1
            yield arr, k - 1, k - 1

        while j <= max_j:
            arr[k] = temp[j]
            j += 1
            k += 1
            yield arr, k - 1, k - 1

    yield from _merge_sort(array, 0, len(array) - 1)


def bitonic_sort_gen(array):
    def _bitonic_sort(arr, low, cnt, up):
        if cnt > 1:
            k = cnt // 2
            yield from _bitonic_sort(arr, low, k, True)
            yield from _bitonic_sort(arr, low + k, k, False)
            yield from _bitonic_merge(arr, low, cnt, up)

    def _bitonic_merge(arr, low, cnt, up):
        if cnt > 1:
            k = cnt // 2
            for i in range(low, low + k):
                if (arr[i] > arr[i + k]) == up:
                    arr[i], arr[i + k] = arr[i + k], arr[i]
                    yield arr, i, i + k
            yield from _bitonic_merge(arr, low, k, up)
            yield from _bitonic_merge(arr, low + k, k, up)

    yield from _bitonic_sort(array, 0, len(array), True)


# Optimized sorting algorithms (generator versions)
def quick_sort_optimized_gen(array):
    def _quick_sort(arr, low, high):
        while low < high:
            if high - low < 16:
                for _ in insertion_sort_gen(arr, low, high):
                    yield _
                break
            else:
                pivot = median_of_three(arr, low, high)
                pi = yield from partition(arr, low, high, pivot)
                if pi - low < high - pi:
                    yield from _quick_sort(arr, low, pi - 1)
                    low = pi + 1
                else:
                    yield from _quick_sort(arr, pi + 1, high)
                    high = pi - 1

    def partition(arr, low, high, pivot):
        i = low - 1
        for j in range(low, high):
            if arr[j] <= pivot:
                i += 1
                arr[i], arr[j] = arr[j], arr[i]
                yield arr, i, j
        arr[i + 1], arr[high] = arr[high], arr[i + 1]
        yield arr, i + 1, high
        return i + 1

    def median_of_three(arr, low, high):
        mid = (low + high) // 2
        if arr[low] > arr[mid]:
            arr[low], arr[mid] = arr[mid], arr[low]
        if arr[low] > arr[high]:
            arr[low], arr[high] = arr[high], arr[low]
        if arr[mid] > arr[high]:
            arr[mid], arr[high] = arr[high], arr[mid]
        arr[mid], arr[high] = arr[high], arr[mid]
        return arr[high]

    def insertion_sort_gen(arr, low, high):
        for i in range(low + 1, high + 1):
            key = arr[i]
            j = i - 1
            while j >= low and arr[j] > key:
                arr[j + 1] = arr[j]
                j -= 1
                yield arr, j + 1, j
            arr[j + 1] = key
            yield arr, j + 1, i

    yield from _quick_sort(array, 0, len(array) - 1)


def merge_sort_optimized_gen(array):
    def _merge_sort(arr, aux, low, high):
        if high - low < 16:  # Use insertion sort for small subarrays
            for _ in insertion_sort_gen(arr, low, high):
                yield _
            return
        mid = (low + high) // 2
        yield from _merge_sort(arr, aux, low, mid)
        yield from _merge_sort(arr, aux, mid + 1, high)
        if arr[mid] <= arr[mid + 1]:  # Skip merge if already sorted
            return
        yield from merge(arr, aux, low, mid, high)

    def merge(arr, aux, low, mid, high):
        for i in range(low, high + 1):
            aux[i] = arr[i]
        i, j, k = low, mid + 1, low
        while i <= mid and j <= high:
            if aux[i] <= aux[j]:
                arr[k] = aux[i]
                i += 1
            else:
                arr[k] = aux[j]
                j += 1
            k += 1
            yield arr, k - 1, j - 1  # Fix: Use j - 1 instead of j + low - 1

        while i <= mid:
            arr[k] = aux[i]
            i += 1
            k += 1
            yield arr, k - 1, k - 1

        while j <= high:
            arr[k] = aux[j]
            j += 1
            k += 1
            yield arr, k - 1, k - 1

    def insertion_sort_gen(arr, low, high):
        for i in range(low + 1, high + 1):
            key = arr[i]
            j = i - 1
            while j >= low and arr[j] > key:
                arr[j + 1] = arr[j]
                j -= 1
                yield arr, j + 1, j
            arr[j + 1] = key
            yield arr, j + 1, i

    aux = array.copy()
    yield from _merge_sort(array, aux, 0, len(array) - 1)


def heap_sort_optimized_gen(array):
    def heapify(arr, n, i):
        while True:
            largest = i
            left = 2 * i + 1
            right = 2 * i + 2
            if left < n and arr[left] > arr[largest]:
                largest = left
            if right < n and arr[right] > arr[largest]:
                largest = right
            if largest == i:
                break
            arr[i], arr[largest] = arr[largest], arr[i]
            yield arr, i, largest
            i = largest

    n = len(array)
    # Build heap (bottom-up)
    for i in range(n // 2 - 1, -1, -1):
        yield from heapify(array, n, i)
    # Extract elements
    for i in range(n - 1, 0, -1):
        array[0], array[i] = array[i], array[0]
        yield array, 0, i
        yield from heapify(array, i, 0)


def bitonic_sort_optimized_gen(array):
    def _bitonic_sort(arr, low, cnt, up):
        if cnt > 1:
            k = cnt // 2
            yield from _bitonic_sort(arr, low, k, True)
            yield from _bitonic_sort(arr, low + k, k, False)
            yield from _bitonic_merge(arr, low, cnt, up)

    def _bitonic_merge(arr, low, cnt, up):
        if cnt > 1:
            k = cnt // 2
            for i in range(low, low + k):
                if (arr[i] > arr[i + k]) == up:
                    arr[i], arr[i + k] = arr[i + k], arr[i]
                    yield arr, i, i + k
            yield from _bitonic_merge(arr, low, k, up)
            yield from _bitonic_merge(arr, low + k, k, up)

    n = len(array)
    # Ensure n is a power of 2
    if not (n & (n - 1)) == 0:
        next_power = 1 << (n - 1).bit_length()
        array.extend([float('inf')] * (next_power - n))
        n = next_power

    yield from _bitonic_sort(array, 0, n, True)

    while len(array) > n:
        array.pop()


class SortingVisualizer:
    def __init__(self, root):
        self.root = root
        self.root.title("Sorting Visualizer")
        self.root.geometry("1200x900")
        self.root.configure(bg="#f0f0f0")

        self.array = []
        self.working_array = []
        self.generator = None
        self.start_time = 0
        self.after_id = None
        self.comparisons = 0
        self.swaps = 0
        self.speed = 50  # Default speed (ms)
        self.theme = "light"  # Default theme

        self.setup_controls()
        self.setup_canvas()
        self.setup_elements_box()
        self.setup_stats_box()
        self.apply_theme()

    def setup_controls(self):
        control_frame = ttk.Frame(self.root, padding="10")
        control_frame.pack(fill=tk.X)

        # Array size input
        ttk.Label(control_frame, text="Array Size:").grid(row=0, column=0, padx=5)
        self.size_entry = ttk.Entry(control_frame, width=10)
        self.size_entry.grid(row=0, column=1, padx=5)

        # Min value input
        ttk.Label(control_frame, text="Min Value:").grid(row=0, column=2, padx=5)
        self.min_entry = ttk.Entry(control_frame, width=10)
        self.min_entry.grid(row=0, column=3, padx=5)
        self.min_entry.insert(0, "-100")

        # Max value input
        ttk.Label(control_frame, text="Max Value:").grid(row=0, column=4, padx=5)
        self.max_entry = ttk.Entry(control_frame, width=10)
        self.max_entry.grid(row=0, column=5, padx=5)
        self.max_entry.insert(0, "100")

        # Allow negative numbers checkbox
        self.neg_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(control_frame, text="Allow Negative Numbers", variable=self.neg_var).grid(row=0, column=6, padx=5)

        # Generate array button
        ttk.Button(control_frame, text="Generate Array", command=self.generate_array).grid(row=0, column=7, padx=5)

        # Sort type selection
        ttk.Label(control_frame, text="Sort Type:").grid(row=1, column=0, padx=5)
        self.sort_var = tk.StringVar()
        sort_combo = ttk.Combobox(control_frame, textvariable=self.sort_var,
                                  values=["Quick Sort", "Merge Sort", "Heap Sort", "Bitonic Sort",
                                          "Optimized Quick Sort", "Optimized Merge Sort", "Optimized Heap Sort", "Optimized Bitonic Sort"])
        sort_combo.grid(row=1, column=1, padx=5)
        sort_combo.current(0)

        # Speed selection
        ttk.Label(control_frame, text="Speed:").grid(row=1, column=2, padx=5)
        self.speed_var = tk.StringVar()
        speed_combo = ttk.Combobox(control_frame, textvariable=self.speed_var,
                                   values=["Slow", "Medium", "Fast"])
        speed_combo.grid(row=1, column=3, padx=5)
        speed_combo.current(1)
        speed_combo.bind("<<ComboboxSelected>>", self.update_speed)

        # Start sorting button
        ttk.Button(control_frame, text="Start Sorting", command=self.start_sorting).grid(row=1, column=4, padx=5)

        # Reset button
        ttk.Button(control_frame, text="Reset", command=self.reset).grid(row=1, column=5, padx=5)

        # Theme selection
        ttk.Label(control_frame, text="Theme:").grid(row=1, column=6, padx=5)
        self.theme_var = tk.StringVar()
        theme_combo = ttk.Combobox(control_frame, textvariable=self.theme_var, values=["Light", "Dark"])
        theme_combo.grid(row=1, column=7, padx=5)
        theme_combo.current(0)
        theme_combo.bind("<<ComboboxSelected>>", self.change_theme)

    def setup_canvas(self):
        self.canvas = tk.Canvas(self.root, width=1100, height=500, bg="white")
        self.canvas.pack(pady=10)

    def setup_elements_box(self):
        elements_frame = ttk.Frame(self.root, padding="10")
        elements_frame.pack(fill=tk.BOTH, expand=True)

        ttk.Label(elements_frame, text="Array Elements:").pack(anchor=tk.W)

        # Text widget with scrollbars
        self.text_scroll_y = ttk.Scrollbar(elements_frame)
        self.text_scroll_x = ttk.Scrollbar(elements_frame, orient=tk.HORIZONTAL)

        self.elements_text = tk.Text(elements_frame, height=4, wrap=tk.NONE,
                                     yscrollcommand=self.text_scroll_y.set,
                                     xscrollcommand=self.text_scroll_x.set,
                                     state=tk.DISABLED)

        self.text_scroll_y.pack(side=tk.RIGHT, fill=tk.Y)
        self.text_scroll_x.pack(side=tk.BOTTOM, fill=tk.X)
        self.elements_text.pack(fill=tk.BOTH, expand=True)

        self.text_scroll_y.config(command=self.elements_text.yview)
        self.text_scroll_x.config(command=self.elements_text.xview)

    def setup_stats_box(self):
        stats_frame = ttk.Frame(self.root, padding="10")
        stats_frame.pack(fill=tk.X)

        self.stats_label = ttk.Label(stats_frame, text="Comparisons: 0 | Swaps: 0 | Time: 0.000s")
        self.stats_label.pack()

    def generate_array(self):
        try:
            size = int(self.size_entry.get())
            min_val = int(self.min_entry.get())
            max_val = int(self.max_entry.get())
            allow_neg = self.neg_var.get()

            if not allow_neg and min_val < 0:
                messagebox.showerror("Error", "Min value cannot be negative when negative numbers are disabled.")
                return

            self.array = [random.randint(min_val, max_val) for _ in range(size)]
            if not allow_neg:
                self.array = [abs(num) for num in self.array]
            self.draw_array()
        except ValueError:
            messagebox.showerror("Error", "Invalid input values")

    def draw_array(self, highlighted_indices=[]):
        self.canvas.delete("all")
        if not self.array:
            return

        max_val = max(abs(num) for num in self.array)
        if max_val == 0:
            max_val = 1

        canvas_height = 500
        canvas_width = 1100
        baseline = canvas_height // 2
        bar_width = canvas_width / len(self.array)

        for i, num in enumerate(self.array):
            x0 = i * bar_width
            x1 = (i + 1) * bar_width
            scaled_height = (abs(num) / max_val) * (baseline - 20)

            if num >= 0:
                y0 = baseline - scaled_height
                y1 = baseline
            else:
                y0 = baseline
                y1 = baseline + scaled_height

            fill_color = "dodger blue"
            if i in highlighted_indices:
                fill_color = "red"
            self.canvas.create_rectangle(x0, y0, x1, y1, fill=fill_color, outline="black")

            if bar_width > 30:
                text_y = baseline - scaled_height - 15 if num >= 0 else baseline + scaled_height + 15
                self.canvas.create_text((x0 + x1) / 2, text_y,
                                       text=str(num), font=('Arial', 10 if bar_width > 50 else 8))

        # Update elements text box
        self.elements_text.config(state=tk.NORMAL)
        self.elements_text.delete(1.0, tk.END)
        elements_str = ", ".join(map(str, self.array))
        self.elements_text.insert(tk.END, elements_str)
        self.elements_text.config(state=tk.DISABLED)
        self.elements_text.xview_moveto(0)

    def start_sorting(self):
        if self.sort_var.get() == 'Bitonic Sort' and not self.is_power_of_two(len(self.array)):
            messagebox.showerror("Error", "Bitonic Sort requires array size to be a power of two")
            return

        self.working_array = self.array.copy()
        self.start_time = time.time()
        self.comparisons = 0
        self.swaps = 0

        sort_type = self.sort_var.get()
        if sort_type == 'Quick Sort':
            self.generator = quick_sort_gen(self.working_array)
        elif sort_type == 'Merge Sort':
            self.generator = merge_sort_gen(self.working_array)
        elif sort_type == 'Heap Sort':
            self.generator = heap_sort_gen(self.working_array)
        elif sort_type == 'Bitonic Sort':
            self.generator = bitonic_sort_gen(self.working_array)
        elif sort_type == 'Optimized Quick Sort':
            self.generator = quick_sort_optimized_gen(self.working_array)
        elif sort_type == 'Optimized Merge Sort':
            self.generator = merge_sort_optimized_gen(self.working_array)
        elif sort_type == 'Optimized Heap Sort':
            self.generator = heap_sort_optimized_gen(self.working_array)
        elif sort_type == 'Optimized Bitonic Sort':
            self.generator = bitonic_sort_optimized_gen(self.working_array)

        self.animate()

    def is_power_of_two(self, n):
        return (n & (n - 1)) == 0 and n != 0

    def animate(self):
        try:
            array_state, idx1, idx2 = next(self.generator)
            self.comparisons += 1

            # Check if a swap occurred
            if idx1 != idx2:  # Swap occurred if indices are different
                self.swaps += 1

            self.array[:] = array_state
            self.draw_array([idx1, idx2])
            elapsed = time.time() - self.start_time
            self.stats_label.config(
                text=f"Comparisons: {self.comparisons} | Swaps: {self.swaps} | Time: {elapsed:.3f}s")
            self.after_id = self.root.after(self.speed, self.animate)
        except StopIteration:
            elapsed = time.time() - self.start_time
            messagebox.showinfo("Sort Complete",
                                f"Time taken: {elapsed:.3f} seconds\nComparisons: {self.comparisons}\nSwaps: {self.swaps}")
            self.after_id = None

    def update_speed(self, event):
        speed_map = {"Slow": 200, "Medium": 50, "Fast": 10}
        self.speed = speed_map[self.speed_var.get()]

    def change_theme(self, event):
        self.theme = self.theme_var.get().lower()
        self.apply_theme()

    def apply_theme(self):
        bg_color = "#f0f0f0" if self.theme == "light" else "#2d2d2d"
        fg_color = "black" if self.theme == "light" else "white"
        self.root.configure(bg=bg_color)
        for widget in self.root.winfo_children():
            if isinstance(widget, ttk.Frame):
                widget.configure(style=f"{self.theme}.TFrame")
            elif isinstance(widget, ttk.Label):
                widget.configure(style=f"{self.theme}.TLabel")
            elif isinstance(widget, ttk.Button):
                widget.configure(style=f"{self.theme}.TButton")
            elif isinstance(widget, ttk.Combobox):
                widget.configure(style=f"{self.theme}.TCombobox")
            elif isinstance(widget, ttk.Checkbutton):
                widget.configure(style=f"{self.theme}.TCheckbutton")
        self.canvas.configure(bg="white" if self.theme == "light" else "#333333")

    def reset(self):
        if self.after_id is not None:
            self.root.after_cancel(self.after_id)
            self.after_id = None

        self.array = []
        self.working_array = []
        self.generator = None
        self.size_entry.delete(0, tk.END)
        self.min_entry.delete(0, tk.END)
        self.max_entry.delete(0, tk.END)
        self.min_entry.insert(0, "-100")
        self.max_entry.insert(0, "100")
        self.canvas.delete("all")
        self.elements_text.config(state=tk.NORMAL)
        self.elements_text.delete(1.0, tk.END)
        self.elements_text.config(state=tk.DISABLED)
        self.stats_label.config(text="Comparisons: 0 | Swaps: 0 | Time: 0.000s")


if __name__ == "__main__":
    root = tk.Tk()
    app = SortingVisualizer(root)
    root.mainloop()