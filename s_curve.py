import math
from turtle import color
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import itertools

pi = math.pi

class SCurve:
    def __init__(self, t_all, S_all):
        t1 = t_all / 2.0
        self.t1 = t1
        self.S1 = S_all / 2.0
        self.vmax = S_all / t1
        w = 2 * pi / t1
        self.w = w
        self.A = self.vmax * w * w / (2 * pi)
        s1 = (2 * self.A * pi * pi) / (w * w * w)
        assert abs(self.S1 - s1) < 0.00000000000001

    def jerk(self, t):
        if t < self.t1:
            c = self.A
        else:
            t -= self.t1
            c = -self.A

        return c* math.sin(self.w * t)

    def acc(self, t):
        if t < self.t1:
            c = self.A
        else:
            t -= self.t1
            c = -self.A

        c_w = c / self.w

        return c_w * (1 - math.cos(self.w * t))

    def vel(self, t):
        global v0

        if t < self.t1:
            c = self.A
            v0 = 0
        else:
            t -= self.t1
            c = -self.A
            v0 = self.vmax

        c_w = c / self.w

        return v0 + c_w * (t - math.sin(self.w * t) / self.w)

    def dist(self, t):
        if t < self.t1:
            c = self.A
            s = 0
        else:
            t -= self.t1
            c = -self.A
            s = self.S1 + self.vmax * t

        c_w = c / self.w

        return s + c_w * (0.5 * t * t + (math.cos(self.w * t) - 1) / (self.w * self.w) )

if __name__ == '__main__':

    cnt = 500
    fig = plt.figure()

    for idx in range(2):
        S_all = 3 if idx == 0 else -3

        sc = SCurve(2, S_all)
        ts = np.linspace(0, 2 * sc.t1, cnt)

        dt = 2 * sc.t1 / cnt

        # 躍度
        ax = fig.add_subplot(4, 2, 1 + idx)
        ax.plot(ts, [sc.jerk(t) for t in ts])
        ax.set_title('躍度', fontname="Meiryo")

        # 加速度
        ax = fig.add_subplot(4, 2, 3 + idx)
        ax.plot(ts, [sc.acc(t) for t in ts], color='blue')
        ax.plot(ts, list(itertools.accumulate([sc.jerk(t) * dt for t in ts])), color='red')
        ax.set_title('加速度', fontname="Meiryo")

        # 速度
        ax = fig.add_subplot(4, 2, 5 + idx)
        ax.plot(ts, [sc.vel(t) for t in ts], color='blue')
        ax.plot(ts, list(itertools.accumulate([sc.acc(t) * dt for t in ts])), color='red')
        ax.set_title('速度', fontname="Meiryo")

        # 距離
        ax = fig.add_subplot(4, 2, 7 + idx)
        # ax.set_ylim(0, 1.1 * 2 * sc.S1)
        ax.plot(ts, [sc.dist(t) for t in ts], color='blue')
        ax.plot(ts, list(itertools.accumulate([sc.vel(t) * dt for t in ts])), color='red')
        ax.set_title('距離', fontname="Meiryo")

    fig.tight_layout()

    plt.show()