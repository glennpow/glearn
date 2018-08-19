import numpy as np
import matplotlib
matplotlib.use("TkAgg")
# matplotlib.use("MacOSX")
# matplotlib.use("Qt4agg")
import matplotlib.pyplot as plt  # noqa


class Dashboard(object):
    def __init__(self, num=None, figsize=None, dpi=None, facecolor=None, edgecolor=None,
                 frameon=True, FigureClass=None,
                 show=True, **kwargs):
        if FigureClass is None:
            FigureClass = matplotlib.figure.Figure

        plt.ion()
        self.figure = plt.figure(num=num, figsize=figsize, dpi=dpi, facecolor=facecolor,
                                 edgecolor=edgecolor, frameon=frameon, FigureClass=FigureClass,
                                 **kwargs)
        # FIXME... raise the window
        # self.figure.canvas.manager.window.raise_()
        # self.figure.canvas.get_tk_widget().focus_force()
        # cfm = plt.get_current_fig_manager()
        # cfm.window.activateWindow()
        # cfm.window.raise_()

        self.axes = {}
        if show:
            self.show()

    def show(self):
        self.figure.show()

    def close(self):
        plt.close(self.figure)

    def plot(self, name, values, xlabel="X", ylabel="Y"):
        x = np.arange(len(values))
        y = values

        if name in self.axes:
            ax = self.axes[name]

            ax.set_xdata(x)
            ax.set_ydata(y)
        else:
            plt.figure(self.figure.number)
            plt.plot(x, y)
            ax = plt.gca()
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)

            self.axes[name] = ax

        plt.draw()
        plt.pause(0.001)
        return ax

    def grid(self, name, values, rows, cols):
        grid = values.reshape(rows, cols)

        if name in self.axes:
            ax = self.axes[name]

            ax.set_data(grid)
        else:
            plt.figure(self.figure.number)
            ax = plt.imshow(grid, cmap='hot', interpolation='none', clim=(0, 1))

            self.axes[name] = ax

        plt.draw()
        plt.pause(0.001)
        return ax
