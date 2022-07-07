"""
Taken from: https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
"""

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def show_landmarks(image, predicted, real=None):
    plt.imshow(image)
    plt.scatter(predicted[0, :], predicted[1, :], s=10, marker=".", c="r")
    if real is not None:
        plt.scatter(real[0, :], real[1, :], s=10, marker=".", c="b")
    plt.pause(0.001)
    plt.show()


def show_face_box(image, box):
    plt.imshow(image)
    ax = plt.gca()
    rect = Rectangle((box[0], box[1]), box[2], box[3], fill=False)
    ax.add_patch(rect)
    plt.pause(0.001)
    plt.show()


def show_img(image):
    plt.imshow(image)
    plt.pause(0.001)
    plt.show()
