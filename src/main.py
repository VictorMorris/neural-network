import random
import threading
import tkinter as tk
from tkinter import ttk
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.ticker import MaxNLocator

from network import Network
from readdata import MnistDataloader


# Locations of data
TRAINING_IMAGES = 'input/train-images.idx3-ubyte'
TRAINING_LABELS = 'input/train-labels.idx1-ubyte'
TEST_IMAGES     = 'input/t10k-images.idx3-ubyte'
TEST_LABELS     = 'input/t10k-labels.idx1-ubyte'


class App:
    def __init__(self, root, x_train, y_train, network, training_data, test_data):
        self.root = root
        self.x_train = x_train
        self.y_train = y_train
        self.network = network
        self.training_data = training_data
        self.test_data = test_data
        self.current_index = random.randint(0, len(self.x_train) - 1)
        self.num_correct = 0
        self.num_guess = 0
        self.is_training = False
        self._graph_dirty = False
        self._lock = threading.Lock()
        self.flash_time = 10  # ms between image flashes during training

        root.title("VictorNet 0.2")
        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)

        self._build_config_frame()
        self._build_main_frame()

        self.config_frame.grid(row=0, column=0)

    # Config screen
    def _build_config_frame(self):
        self.config_frame = ttk.Frame(self.root, padding="60 60 60 60")
        self.config_frame.columnconfigure(0, weight=1)
        self.config_frame.columnconfigure(1, weight=1)

        ttk.Label(self.config_frame, text="VictorNet 0.2", font=("Arial", 24, "bold")).grid(row=0, column=0, columnspan=2, pady=(0, 40))

        fields = [
            ("Epochs",        "30",  "epochs_var"),
            ("Batch Size",    "10",  "batch_var"),
            ("Learning Rate", "3.0", "lr_var"),
        ]
        for i, (label, default, attr) in enumerate(fields, start=1):
            ttk.Label(self.config_frame, text=f"{label}:", font=("Arial", 12)).grid(
                row=i, column=0, sticky=tk.E, padx=10, pady=8)
            var = tk.StringVar(value=default)
            setattr(self, attr, var)
            ttk.Entry(self.config_frame, textvariable=var, width=10).grid(
                row=i, column=1, sticky=tk.W, pady=8)

        tk.Button(self.config_frame, text="Start Training", font=("Arial", 14),
                  command=self._start_training).grid(row=4, column=0, columnspan=2, pady=40)

    # Main training/results screen
    def _build_main_frame(self):
        self.mainframe = ttk.Frame(self.root, padding="10 10 10 10")
        self.mainframe.columnconfigure(0, weight=1)

        self.fig_img, self.ax_img = plt.subplots()
        self.show_image()
        self.canvas_img = FigureCanvasTkAgg(self.fig_img, master=self.mainframe)
        self.canvas_img.draw()

        self.fig_graph, self.ax_graph = plt.subplots()
        self.canvas_graph = FigureCanvasTkAgg(self.fig_graph, master=self.mainframe)
        self.show_graph()
        self.canvas_graph.draw()

        self.buttons_frame = ttk.Frame(self.mainframe)
        tk.Button(self.buttons_frame, text="New Image", command=self.new_image).grid(
            row=0, column=0, padx=5)
        self.buttons_frame.columnconfigure(0, weight=1)

        self.training_label = tk.Label(self.mainframe, text="Network Training...", font=("Arial", 14))
        self.prediction_label = tk.Label(self.mainframe, text="", font=("Arial", 14))
        self.result_label = tk.Label(self.mainframe, text="", font=("Arial", 14))

        self.canvas_img.get_tk_widget().grid(row=0, column=0)
        self.canvas_graph.get_tk_widget().grid(row=0, column=1)
        self.training_label.grid(row=1, column=0, pady=10)

    def _start_training(self):
        try:
            epochs = int(self.epochs_var.get())
            batch_size = int(self.batch_var.get())
            lr = float(self.lr_var.get())
        except ValueError:
            return

        self.is_training = True
        self.config_frame.grid_remove()
        self.mainframe.grid(row=0, column=0)
        self.root.after(50, self._training_loop)

        def run_training():
            self.network.SGD(
                self.training_data, epochs, batch_size, lr,
                test_data=self.test_data,
                epoch_callback=self.on_epoch_complete,
            )
            self.root.after(0, self.training_done)

        threading.Thread(target=run_training, daemon=True).start()

    # Image / graph helpers
    def show_image(self):
        image = self.x_train[self.current_index]
        label = self.y_train[self.current_index]
        self.ax_img.clear()
        self.ax_img.imshow(image.reshape(28, 28), cmap=plt.cm.gray)
        if label != '':
            self.ax_img.set_title(f"Label: {label}", fontsize=15)

    def show_graph(self):
        self.ax_graph.clear()
        self.ax_graph.set_title("Error Rate per Epoch")
        self.ax_graph.set_ylabel("Error Rate")
        self.ax_graph.set_xlabel("Epoch")
        self.ax_graph.xaxis.set_major_locator(MaxNLocator(integer=True))
        errors = self.network.test_errors
        epochs = list(range(1, len(errors) + 1))
        self.ax_graph.set_xlim(0.5, max(len(errors), 2) + 0.5)
        self.ax_graph.plot(epochs, errors)

    # Training callbacks / loop
    def on_epoch_complete(self, _test_error):
        """Called from the training thread after each epoch."""
        with self._lock:
            self._graph_dirty = True

    def training_done(self):
        """Called on the main thread when training finishes."""
        self.is_training = False
        self.show_graph()
        self.canvas_graph.draw()
        self.training_label.grid_remove()
        self.buttons_frame.grid(row=1, column=0, pady=5)
        self.prediction_label.grid(row=2, column=0, pady=5)
        self.result_label.grid(row=3, column=0, pady=5)
        self.predict()

    def _training_loop(self):
        """Runs on the main thread via after(); cycles images and refreshes the graph."""
        self.current_index = random.randint(0, len(self.x_train) - 1)
        self.show_image()
        self.canvas_img.draw()

        with self._lock:
            dirty = self._graph_dirty
            self._graph_dirty = False
        if dirty:
            self.show_graph()
            self.canvas_graph.draw()

        if self.is_training:
            self.root.after(self.flash_time, self._training_loop)

    # User interactions
    def new_image(self):
        self.current_index = random.randint(0, len(self.x_train) - 1)
        self.show_image()
        self.canvas_img.draw()
        self.predict()

    def predict(self):
        activations = self.network.feed_forward(self.x_train[self.current_index])
        print(activations)
        guess = list(activations).index(max(activations))
        if guess == self.y_train[self.current_index]:
            self.num_correct += 1
        self.num_guess += 1
        self.prediction_label.config(text=f"Network thinks: {guess}")
        self.result_label.config(text=f"{self.num_correct}/{self.num_guess}")


def _quit(root):
    root.quit()
    root.destroy()


def main():
    loader = MnistDataloader(TRAINING_IMAGES, TRAINING_LABELS, TEST_IMAGES, TEST_LABELS)
    (x_train, y_train), (x_test, y_test) = loader.load_data()

    network = Network([784, 16, 16, 10])

    def to_one_hot(label):
        v = np.zeros((10, 1))
        v[label] = 1.0
        return v

    training_data = [(x, to_one_hot(y)) for x, y in zip(x_train, y_train)]
    test_data = [(x, to_one_hot(y)) for x, y in zip(x_test, y_test)]

    root = tk.Tk()
    root.protocol("WM_DELETE_WINDOW", lambda: _quit(root))
    App(root, x_train, y_train, network, training_data, test_data)
    root.mainloop()


if __name__ == '__main__':
    main()
