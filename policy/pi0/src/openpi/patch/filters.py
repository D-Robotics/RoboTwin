from scipy.signal import butter
from scipy.signal import filtfilt
from scipy.signal import firwin
from scipy.signal import lfilter

import numpy as np

class NoFilter:
    def __init__(self, *args, **kwargs):
        # 重写__new__，创建实例时直接返回None（跳过对象创建）
        print("NoFilter")

    def reset(self):
        pass

    def filter(self, data):
        return data


class ZeroPhaseFTR:
    def __init__(self, numtaps=11, cutoff=3, fs=50, channels=14):
        self.b = firwin(numtaps, cutoff, fs=fs)
        self.channels = channels
        print("ZeroPhase")

    def reset(self):
        pass

    def filter(self, data):
        """
        对数据进行前向+反向零相位滤波（离线）
        :param data: shape (T, channels)
        :return: 滤波后的数据
        """
        y = np.zeros_like(data)
        for ch in range(self.channels):
            y[:, ch] = filtfilt(self.b, 1, data[:, ch])
        return y


class FIR:
    def __init__(self, numtaps=11, cutoff=3, fs=50, channels=14):
        """
        在线 FIR 滤波器
        :param numtaps: FIR 长度（最好为奇数）
        :param cutoff: 截止频率 (Hz)
        :param fs: 采样率 (Hz)
        :param channels: 数据通道数（例如 14 维向量）
        """
        self.numtaps = numtaps
        self.b = firwin(numtaps, cutoff, fs=fs)
        self.channels = channels
        self.zi = np.zeros((numtaps - 1, channels))  # 每列保存一个通道的历史
        print("FIR")

    def reset(self):
        """清除历史缓存"""
        self.zi[:] = 0

    def filter(self, data):
        """
        对数据进行在线 FIR 滤波
        :param data: shape (T, channels)
        :return: 滤波后的数据，shape (T, channels)
        """
        if data.shape[1] != self.channels:
            raise ValueError(f"数据通道数 {data.shape[1]} 与滤波器设置 {self.channels} 不一致")

        y = np.zeros_like(data)
        for ch in range(self.channels):
            y[:, ch], self.zi[:, ch] = lfilter(self.b, 1, data[:, ch], zi=self.zi[:, ch])
        return y


class MultiChannelButterworth:
    def __init__(self, cutoff, fs, channels=14, order=2):
        self.b, self.a = butter(order, cutoff / (0.5 * fs), btype="low")
        self.order = order
        self.channels = channels
        self.x_hist = np.zeros((len(self.b), channels))
        self.y_hist = np.zeros((len(self.a), channels))
        print("Butter")

    def reset(self):
        self.x_hist = np.zeros((len(self.b), self.channels))
        self.y_hist = np.zeros((len(self.a), self.channels))

    def filter(self, arr):
        filtered = np.zeros_like(arr)
        for i in range(arr.shape[0]):
            filtered[i, :] = self.filter_impl(arr[i, :])
        return filtered

    def filter_impl(self, x):
        x = np.asarray(x)
        assert x.shape == (self.channels,), f"Expected shape ({self.channels},), got {x.shape}"
        # Shift history

        self.x_hist[1:] = self.x_hist[:-1]
        self.x_hist[0] = x

        self.y_hist[1:] = self.y_hist[:-1]

        # Compute output per channel
        y = (self.b[:, None] * self.x_hist).sum(axis=0) - (self.a[1:, None] * self.y_hist[1:]).sum(axis=0)
        y /= self.a[0]

        self.y_hist[0] = y

        return y

