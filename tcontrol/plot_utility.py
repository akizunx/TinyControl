import math

import matplotlib.widgets as widgets
import numpy as np
from matplotlib import pyplot as plt

from tcontrol import pzmap

__all__ = ['grid', 'figure', 'show', "AnnotatedPoint", "AttachedCursor",
           'plot_nyquist', 'plot_response_curve', 'plot_bode', 'plot_pzmap', 'plot_rlocus']


def show(*args, **kwargs):
    return plt.show(*args, **kwargs)


def grid(b=None, which='major', axis='both', **kwargs):
    return plt.grid(b, which, axis, **kwargs)


def figure(num=None, figsize=None, dpi=None, facecolor=None, edgecolor=None, frameon=True,
           FigureClass=plt.Figure, **kwargs):
    return plt.figure(num, figsize, dpi, facecolor, edgecolor, frameon, FigureClass,
                      **kwargs)


class AnnotatedPoint(object):
    def __init__(self, ax, fig):
        """

        :param ax:
        :type ax: matplotlib.axes.Axes
        :param fig:
        :type fig: matplotlib.figure.Figure
        """
        self.ax = ax
        self.fig = fig
        self.anno = self.init_annotate()
        self.dot = None
        self.cig = []

    def __call__(self, *args, **kwargs):
        raise NotImplementedError("should be implemented in subclass.")

    def init_annotate(self):
        props = {"boxstyle": "square", "facecolor": "white", "alpha": 0.9}
        arrowprops = {"arrowstyle": '->', "connectionstyle": 'arc3, rad=0'}
        anno = self.ax.annotate('', xytext=(0.25, 0.25),
                                xy=(0, 0), bbox=props, arrowprops=arrowprops)

        return anno

    def connect(self, *events):
        for event in events:
            cig = self.fig.canvas.mpl_connect(event, self)
            self.cig.append(cig)

    def disconnect(self):
        for i in self.cig:
            self.fig.canvas.mpl_disconnect(i)


class AttachedCursor(widgets.Cursor):
    def __init__(self, ax, fig, **lineprops):
        super().__init__(ax, horizOn=True, vertOn=True, useblit=True,
                         **lineprops)
        self.ax = ax
        self.fig = fig
        self.fig = self.ax.figure
        self.cig = []
        self.ax.set_picker(True)

    def __call__(self, event):
        raise NotImplementedError("should be implemented in subclass.")

    def connect(self, *events):
        for event in events:
            cig = self.fig.canvas.mpl_connect(event, self)
            self.cig.append(cig)

    def disconnect(self):
        for i in self.cig:
            self.fig.canvas.mpl_disconnect(i)


def plot_response_curve(y, t, title, continuous=True):
    fig = plt.figure()
    axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    axes.set_xlabel('t/s')
    axes.set_ylabel('Amplitude')
    axes.set_title(title)
    axes.axvline(x=0, color='black')
    axes.axhline(y=0, color='black')
    if continuous:
        axes.plot(t, y)
    else:
        axes.step(t, y, where='post')
    axes.grid()
    plt.show()


def plot_rlocus(kwargs, roots, system):
    fig, ax = plt.subplots()
    ax.axvline(x=0, color='black')
    ax.axhline(y=0, color='black')
    if 'xlim' in kwargs.keys():
        ax.set_xlim(*kwargs['xlim'])
    if 'ylim' in kwargs.keys():
        ax.set_ylim(*kwargs['xlim'])
    line = []
    for r in roots:
        l, *_ = ax.plot(r.real, r.imag, picker=2)
        line.append(l)
    p, z = pzmap(system, plot=False)
    ax.scatter(np.real(z), np.imag(z), s=50, marker='o', color='#069af3')
    ax.scatter(np.real(p), np.imag(p), s=50, marker='x', color='#fdaa48')
    ax.grid()
    plt.title('Root Locus')
    cursor = RlocusAttachedCursor(ax, fig, line, sys_=system, linestyle='--')
    cursor.connect_event("pick_event", cursor)
    plt.show()
    cursor.disconnect()


class RlocusAnnotatedPoint(AnnotatedPoint):
    def __init__(self, ax, fig, sys_):
        super().__init__(ax, fig)
        self.ax.autoscale(False)
        self.sys_ = sys_
        self.anno.remove()
        self.anno = None

    def __call__(self, event):
        """

        :param event: A matplotlib event
        :type event: matplotlib.back_end.Event
        :return: None
        :rtype: None
        """
        if event.name == "button_press_event":
            self.handle_click(event)
            event.canvas.draw()
        else:
            return None

    def handle_click(self, event):
        """

        :param event: A matplotlib event
        :type event: matplotlib.back_end.Event
        :return: None
        :rtype: None
        """
        if event.inaxes is None:
            return None
        if self.anno is None:
            self.anno = self.init_annotate()

        s = complex(event.xdata, event.ydata)
        num = np.abs(self.sys_.pole() - s)
        den = np.abs(self.sys_.zero() - s)
        k = np.prod(num) / np.prod(den)

        if s.imag >= 0:
            text_str = "$K={0:.5f}$\n$s={1:.5f}+{2:.5f}j$".format(k, s.real, s.imag)
        else:
            text_str = "$K={0:.5f}$\n$s={1:.5f}{2:.5f}j$".format(k, s.real, s.imag)
        self.anno.xy = (event.xdata, event.ydata)
        self.anno.set_text(text_str)
        self.anno.set_x(event.xdata + 0.2)
        self.anno.set_y(event.ydata + 0.2)

        if self.dot is not None:
            self.dot.remove()
        self.dot = self.ax.scatter(event.xdata, event.ydata, marker='+', color='r', s=62)


class RlocusAttachedCursor(AttachedCursor):
    def __init__(self, ax, fig, line, *, sys_, **lineprops):
        super().__init__(ax, fig, **lineprops)
        self.ap = RlocusAnnotatedPoint(self.ax, self.ax.figure, sys_)
        self.line = line

    def __call__(self, event):
        if event.name == "pick_event":
            if event.artist not in self.line:
                return None
            self.ap(event.mouseevent)
        else:
            return None


def plot_nyquist(result):
    plt.axvline(x=0, color='black')
    plt.axhline(y=0, color='black')
    plt.plot(result.real, result.imag, '-', color='#069af3')
    plt.plot(result.real, -result.imag, '--', color='#fdaa48')
    arrow_pos = int(math.log(result.shape[0])) * 2 + 1
    x1, x2 = np.real(result[arrow_pos]), np.real(result[arrow_pos + 1])
    y1, y2 = np.imag(result[arrow_pos]), np.imag(result[arrow_pos + 1])
    dx = x2 - x1
    dy = y2 - y1
    plt.arrow(x1, -y1, -dx, dy, head_width=0.04, color='#fdaa48')
    plt.arrow(x1, y1, dx, dy, head_width=0.04, color='#069af3')
    plt.scatter(-1, 0, s=30, color='r', marker='P')
    plt.grid()
    plt.title("Nyquist Plot")
    plt.xlabel('Real Axis')
    plt.ylabel('Imag Axis')
    plt.show()


def plot_bode(A, omega, phi):
    plt.title("Bode Diagram")
    ax1 = plt.subplot(2, 1, 1)
    plt.axvline(x=0, color='black')
    plt.axhline(y=0, color='black')
    y_range = [i * 20 for i in range(int(min(A)) // 20 - 1, int(max(A)) // 20 + 2)]
    plt.yticks(y_range)
    plt.plot(omega.imag, A, '-', color='#069af3')
    plt.xscale('log')
    plt.grid(which='both')
    plt.ylabel('Magnitude/dB')
    plt.subplot(2, 1, 2, sharex=ax1)
    plt.axvline(x=0, color='black')
    plt.axhline(y=0, color='black')
    plt.xscale('log')
    y_range = [i * 45 for i in range(int(min(A)) // 45 - 1, int(max(A)) // 45 + 2)]
    plt.yticks(y_range)
    plt.plot(omega.imag, phi, '-', color='#069af3')
    plt.grid(which='both')
    plt.ylabel('Phase/deg')
    plt.xlabel('Frequency/(rad/s)')
    plt.show()


def plot_pzmap(pole, sys_, title, zero):
    if sys_.is_dtime:
        x = np.linspace(-1, 1, 100)
        y = np.sqrt(1 - x ** 2)
        plt.plot(x, y, '--', color='#929591')
        plt.plot(x, -y, '--', color='#929591')
    if zero.shape[0]:
        l1 = plt.scatter(np.real(zero), np.imag(zero), s=30, marker='o',
                         color='#069af3')
        plt.legend([l1], ['zero'])
    if pole.shape[0]:
        l2 = plt.scatter(np.real(pole), np.imag(pole), s=30, marker='x',
                         color='#fdaa48')
        plt.legend([l2], ['pole'])
    plt.grid()
    plt.axvline(x=0, color='black')
    plt.axhline(y=0, color='black')
    plt.xlabel('Real Axis')
    plt.ylabel('Imag Axis')
    plt.title(title)
    plt.show()
