import random
import tkinter as tk
from tkinter import ttk
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from network import Network
from readdata import MnistDataloader


# Locations of data
TRAINING_IMAGES = 'input/train-images.idx3-ubyte'
TRAINING_LABELS = 'input/train-labels.idx1-ubyte'
TEST_IMAGES     = 'input/t10k-images.idx3-ubyte'
TEST_LABELS     = 'input/t10k-labels.idx1-ubyte'


class App:
    def __init__(self, root, x_train, y_train, network):
        self.root = root
        self.x_train = x_train
        self.y_train = y_train
        self.network = network
        self.current_index = random.randint(0, len(self.x_train) - 1)
        self.num_correct = 0
        self.num_guess = 0

        root.title("VictorNet 0.1")
        root.geometry("450x700")

        mainframe = ttk.Frame(root, padding="10 10 10 10")
        mainframe.grid(column=0, row=0, sticky=(tk.N, tk.W, tk.E, tk.S))
        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)
        mainframe.columnconfigure(0, weight=1)

        self.fig, self.ax = plt.subplots()
        self.show_image()

        self.canvas = FigureCanvasTkAgg(self.fig, master=mainframe)
        self.canvas.draw()
        self.canvas.get_tk_widget().grid(row=0, column=0)

        buttons_frame = ttk.Frame(mainframe)
        buttons_frame.grid(row=1, column=0, pady=5)

        tk.Button(buttons_frame, text="New Image", command=self.new_image).grid(row=0, column=0, padx=5, sticky="ew")
        buttons_frame.columnconfigure(0, weight=1)

        self.prediction_label = tk.Label(mainframe, text="", font=("Arial", 14))
        self.prediction_label.grid(row=2, column=0, pady=5)

        self.result_label = tk.Label(mainframe, text="", font=("Arial", 14))
        self.result_label.grid(row=3, column=0, pady=5)

        self.predict()

    def show_image(self):
        image = self.x_train[self.current_index]
        label = self.y_train[self.current_index]
        self.ax.clear()
        self.ax.imshow(image.reshape(28, 28), cmap=plt.cm.gray)
        if label != '':
            self.ax.set_title(f"Label: {label}", fontsize=15)

    def new_image(self):
        self.current_index = random.randint(0, len(self.x_train) - 1)
        self.show_image()
        self.canvas.draw()
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
    """Function to be called when the window is closed."""
    root.quit()     # Stops the mainloop
    root.destroy()  # Destroys the window and all widgets

def main():
    loader = MnistDataloader(TRAINING_IMAGES, TRAINING_LABELS, TEST_IMAGES, TEST_LABELS)
    (x_train, y_train), (x_test, y_test) = loader.load_data()

    network = Network([784, 16, 16, 10])

    def to_one_hot(label):
        v = np.zeros((10,1))
        v[label] = 1.0
        return v
    training_data = [(x, to_one_hot(y)) for x, y in zip(x_train, y_train)]
    test_data = [(x, to_one_hot(y)) for x, y in zip(x_test, y_test)]
    network.SGD(training_data, 30, 10, 3.0, test_data=test_data)

    root = tk.Tk()
    root.protocol("WM_DELETE_WINDOW", lambda: _quit(root))
    App(root, x_train, y_train, network)
    root.mainloop()


if __name__ == '__main__':
    main()
