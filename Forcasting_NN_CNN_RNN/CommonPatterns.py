#https://classroom.udacity.com/courses/ud187/lessons/6d543d5c-6b18-4ecf-9f0f-3fd034acd2cc/concepts/6a5d37fc-553c-4def-997e-da678c3054c4
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import matplotlib.pyplot as plt

def plot_series(time, series, format="-", start=0, end=None, label=None):
    plt.plot(time[start:end], series[start:end], format, label=label)
    plt.xlabel("Time")
    plt.ylabel("value")
    if label:
        plt.legend(fontsize=14)
    plt.grid(True)

def trend(time,slope=0):
    return slope*time

time = np.arange(4*365+1)
baseline = 10
series = baseline + trend(time,0.1)

plt.figure(figsize=(10,6))
plot_series(time,series)
plt.title("Up Trend")
print(time,"time")
print(series,"series")

def seasonal_pattern(season_time):
    """Just an arbitrary pattern, you can change it if you wish"""
    return np.where(season_time<0.4,
                    np.cos(season_time*2*np.pi),
                    1/np.exp(3*season_time))

def seasononality(time, period, amplitude=1, phase=0):
    '''Repeats the same pattern at each period'''
    season_time = ((time + phase) % period) / period
    return amplitude*seasonal_pattern(season_time)

amplitude = 40
series = seasononality(time, period=365, amplitude=amplitude)

plt.figure()
plot_series(time, series)
plt.title("Seasonal Pattern")

slope = 0.05
series = baseline + trend(time, slope) + seasononality(time, period=365, amplitude=amplitude)

plt.figure()
plot_series(time, series)
plt.title("Up Trend + Seasonal Pattern")

def white_noise(time, noise_level=1, seed=None):
    rnd = np.random.RandomState(seed)
    return rnd.randn(len(time))*noise_level

noise_level = 5
noise = white_noise(time,noise_level, seed=42)

plt.figure()
plot_series(time, noise)
plt.title("White Noise")

series = series + noise

plt.figure()
plot_series(time, series)
plt.title("White Noise + Seasonal Pattern + Up Trend")
plt.show()
