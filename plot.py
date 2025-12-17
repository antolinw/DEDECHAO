import numpy as np
import matplotlib.pyplot as plt

def plot_surface(field, title="", cmap="viridis"):
    N = field.shape[0]
    x = np.linspace(-1, 1, N)
    y = np.linspace(-1, 1, N)
    X, Y = np.meshgrid(x, y)

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")

    surf = ax.plot_surface(X, Y, field, cmap=cmap)
    fig.colorbar(surf)
    ax.set_title(title)

    plt.show()

    plot_img(field, title=title, cmap=cmap)

def plot_img(field, title="", cmap="viridis"):
    plt.imshow(field)
    plt.title(title)
    plt.colorbar()
    plt.show()