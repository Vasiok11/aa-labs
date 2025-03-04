import tkinter as tk
from tkinter import ttk, messagebox
import random
import time


# Sorting algorithms implementations
def quick_sort_gen(array):
    def _quick_sort(arr, low, high):
        if low < high:
            pivot = arr[high]
            i = low - 1
            for j in range(low, high):
                if arr[j] <= pivot:
                    i += 1
                    arr[i], arr[j] = arr[j], arr[i]
                    yield arr
            arr[i + 1], arr[high] = arr[high], arr[i + 1]
            yield arr
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
            yield arr
            yield from heapify(arr, n, largest)

    n = len(array)
    for i in range(n // 2 - 1, -1, -1):
        yield from heapify(array, n, i)

    for i in range(n - 1, 0, -1):
        array[0], array[i] = array[i], array[0]
        yield array
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
            yield arr

        while i <= max_i:
            arr[k] = temp[i]
            i += 1
            k += 1
            yield arr

        while j <= max_j:
            arr[k] = temp[j]
            j += 1
            k += 1
            yield arr

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
                    yield arr
            yield from _bitonic_merge(arr, low, k, up)
            yield from _bitonic_merge(arr, low + k, k, up)

    yield from _bitonic_sort(array, 0, len(array), True)


class SortingVisualizer:
    def __init__(self, root):
        self.root = root
        self.root.title("Sorting Visualizer")
        self.root.geometry("1000x900")

        self.array = []
        self.working_array = []
        self.generator = None
        self.start_time = 0
        self.after_id = None

        self.setup_controls()
        self.setup_canvas()
        self.setup_elements_box()

    def setup_controls(self):
        control_frame = ttk.Frame(self.root)
        control_frame.pack(pady=10)

        # Array size input
        ttk.Label(control_frame, text="Array Size:").grid(row=0, column=0, padx=5)
        self.size_entry = ttk.Entry(control_frame, width=10)
        self.size_entry.grid(row=0, column=1, padx=5)

        # Generate array button
        ttk.Button(control_frame, text="Generate Array",
                   command=self.generate_array).grid(row=0, column=2, padx=5)

        # Sort type selection
        ttk.Label(control_frame, text="Sort Type:").grid(row=0, column=3, padx=5)
        self.sort_var = tk.StringVar()
        sort_combo = ttk.Combobox(control_frame, textvariable=self.sort_var,
                                  values=["Quick Sort", "Merge Sort", "Heap Sort", "Bitonic Sort"])
        sort_combo.grid(row=0, column=4, padx=5)
        sort_combo.current(0)

        # Start sorting button
        ttk.Button(control_frame, text="Start Sorting",
                   command=self.start_sorting).grid(row=0, column=5, padx=5)

        # Reset button
        ttk.Button(control_frame, text="Reset",
                   command=self.reset).grid(row=0, column=6, padx=5)

    def setup_canvas(self):
        self.canvas = tk.Canvas(self.root, width=900, height=500, bg="white")
        self.canvas.pack(pady=10)

    def setup_elements_box(self):
        elements_frame = ttk.Frame(self.root)
        elements_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

        ttk.Label(elements_frame, text="Array Elements:").pack(anchor=tk.W)

        # Text widget with scrollbars
        self.text_scroll_y = ttk.Scrollbar(elements_frame)
        self.text_scroll_x = ttk.Scrollbar(elements_frame, orient=tk.HORIZONTAL)

        self.elements_text = tk.Text(elements_frame,
                                     height=4,
                                     wrap=tk.NONE,
                                     yscrollcommand=self.text_scroll_y.set,
                                     xscrollcommand=self.text_scroll_x.set,
                                     state=tk.DISABLED)

        self.text_scroll_y.pack(side=tk.RIGHT, fill=tk.Y)
        self.text_scroll_x.pack(side=tk.BOTTOM, fill=tk.X)
        self.elements_text.pack(fill=tk.BOTH, expand=True)

        self.text_scroll_y.config(command=self.elements_text.yview)
        self.text_scroll_x.config(command=self.elements_text.xview)

    def generate_array(self):
        try:
            size = int(self.size_entry.get())
            self.array = [random.randint(-100, 100) for _ in range(size)]
            self.draw_array()
        except ValueError:
            messagebox.showerror("Error", "Invalid array size")

    def draw_array(self):
        # Update visualization bars
        self.canvas.delete("all")
        if not self.array:
            return

        max_val = max(abs(num) for num in self.array)
        if max_val == 0:
            max_val = 1

        canvas_height = 500
        canvas_width = 900
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

            self.canvas.create_rectangle(x0, y0, x1, y1, fill="dodger blue", outline="black")

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
        self.elements_text.xview_moveto(0)  # Reset horizontal scroll

    def start_sorting(self):
        if self.sort_var.get() == 'Bitonic Sort' and not self.is_power_of_two(len(self.array)):
            messagebox.showerror("Error", "Bitonic Sort requires array size to be a power of two")
            return

        self.working_array = self.array.copy()
        self.start_time = time.time()

        sort_type = self.sort_var.get()
        if sort_type == 'Quick Sort':
            self.generator = quick_sort_gen(self.working_array)
        elif sort_type == 'Merge Sort':
            self.generator = merge_sort_gen(self.working_array)
        elif sort_type == 'Heap Sort':
            self.generator = heap_sort_gen(self.working_array)
        elif sort_type == 'Bitonic Sort':
            self.generator = bitonic_sort_gen(self.working_array)

        self.animate()

    def is_power_of_two(self, n):
        return (n & (n - 1)) == 0 and n != 0

    def animate(self):
        try:
            next(self.generator)
            self.array[:] = self.working_array.copy()
            self.draw_array()
            self.after_id = self.root.after(50, self.animate)
        except StopIteration:
            elapsed = time.time() - self.start_time
            messagebox.showinfo("Sort Complete", f"Time taken: {elapsed:.3f} seconds")
            self.after_id = None

    def reset(self):
        if self.after_id is not None:
            self.root.after_cancel(self.after_id)
            self.after_id = None

        self.array = []
        self.working_array = []
        self.generator = None
        self.size_entry.delete(0, tk.END)
        self.canvas.delete("all")
        self.elements_text.config(state=tk.NORMAL)
        self.elements_text.delete(1.0, tk.END)
        self.elements_text.config(state=tk.DISABLED)


if __name__ == "__main__":
    root = tk.Tk()
    app = SortingVisualizer(root)
    root.mainloop()