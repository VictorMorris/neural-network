from network import Network
from readdata import MnistDataloader
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import random


training_images_filepath = 'input/train-images.idx3-ubyte'
training_labels_filepath = 'input/train-labels.idx1-ubyte'
test_images_filepath = 'input/t10k-images.idx3-ubyte'
test_labels_filepath = 'input/t10k-labels.idx1-ubyte'

mnist_dataloader = MnistDataloader(training_images_filepath, training_labels_filepath, test_images_filepath, test_labels_filepath)
(x_train, y_train), (x_test, y_test) = mnist_dataloader.load_data()

# Initialize a network with 784 input nodes and 10 output noes
network = Network([784, 16, 16, 10])

# Current image
current_index = [500]

def show_image(r, ax):
    image = x_train[r]
    image_label = y_train[r]
    # Clear previous image
    ax.clear()
    ax.imshow(image.reshape(28,28), cmap=plt.cm.gray)
    if (image_label != ''):
        ax.set_title(f"This is a {image_label}", fontsize = 15)

def new_image():
    current_index[0] = random.randint(0, len(x_train) - 1)
    show_image(current_index[0], ax)
    canvas.draw()
    predict()

def predict():
    activations = network.feed_forward(x_train[current_index[0]])
    print(activations)
    guess = list(activations).index(max(activations))
    prediction_label.config(text=f"Network thinks: {guess}")

root = tk.Tk()
root.title("GPT1000")
root.geometry("450x700")

mainframe = ttk.Frame(root, padding="10 10 10 10")
mainframe.grid(column=0, row=0, sticky=(tk.N, tk.W, tk.E, tk.S))
root.columnconfigure(0, weight=1)
root.rowconfigure(0, weight=1)
mainframe.columnconfigure(0, weight=1)

fig, ax = plt.subplots()
show_image(current_index[0], ax)

canvas = FigureCanvasTkAgg(fig, master=mainframe)
canvas.draw()
canvas.get_tk_widget().grid(row=0, column=0)

buttons_frame = ttk.Frame(mainframe)
buttons_frame.grid(row=1, column=0, pady=5)

new_img_btn = tk.Button(buttons_frame, text="New Image", command=new_image)
new_img_btn.grid(row=0, column=0, padx=5)

prediction_label = tk.Label(mainframe, text="", font=("Arial", 14))
prediction_label.grid(row=2, column=0, pady=5)

tk.mainloop()

