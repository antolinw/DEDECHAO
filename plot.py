import numpy as np
import matplotlib.pyplot as plt

def plot_surface(field, title="", cmap="viridis", pause=1.0):
    N = field.shape[0]
    x = np.linspace(-1, 1, N)
    y = np.linspace(-1, 1, N)
    X, Y = np.meshgrid(x, y)

    # plot_img(field, title=title, cmap=cmap)

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")

    surf = ax.plot_surface(X, Y, field, cmap=cmap)
    fig.colorbar(surf)
    ax.set_title(title)

    plt.show(block=False)
    plt.pause(pause)
    plt.close()

def plot_img(field, title="", cmap="viridis", pause=1.0, cax = None):

    if cax is None:
        vmax = np.max(field.flatten())
        vmin = np.min(field.flatten())
    else:
        vmax = cax
        vmin = -cax
    plt.imshow(field, vmax=vmax, vmin=vmin)
    plt.title(title)
    plt.colorbar()
    plt.show(block=False)
    plt.pause(pause)
    plt.close()