import matplotlib.pyplot as plt
import numpy as np


def set_plot(x_label, y_label, title, dimensions):
    fig, ax = plt.subplots(figsize=dimensions)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    return fig, ax


def update(plot_data, x, y):
    fig, ax = plot_data
    (graph,) = ax.plot(x, y, color="g")
    ax.set_xlim(x[0], x[-1])
    ax.set_ylim(0, 10)
    return (graph,)
