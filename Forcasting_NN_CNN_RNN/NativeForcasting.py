#https://classroom.udacity.com/courses/ud187/lessons/6d543d5c-6b18-4ecf-9f0f-3fd034acd2cc/concepts/5bee872d-d59c-419a-923b-24969fcfeb09
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

def seasonal_pattern(season_time):
    """Just an arbitrary pattern, you can change it if you wish"""
    return np.where(season_time<0.4,
                    np.cos(season_time*2*np.pi),
                    1/np.exp(3*season_time))

def seasonality(time, period, amplitude=1, phase=0):
    '''Repeats the same pattern at each period'''
    season_time = ((time + phase) % period) / period
    return amplitude*seasonal_pattern(season_time)

def white_noise(time, noise_level=1, seed=None):
    rnd = np.random.RandomState(seed)
    return rnd.randn(len(time))*noise_level

time  = np.arange(4*365+1)

slope = 0.05
baseline = 10
amplitude = 40
series = baseline + trend(time, slope) + seasonality(time, period=365, amplitude=amplitude)

noise_level = 5
noise = white_noise(time, noise_level, seed=42)

series = series + noise

plt.figure()
plot_series(time, series)
plt.title("White Noise + Seasonality + Up Trend")

split_time = 1000
time_train = time[:split_time]
x_train = series[:split_time]
time_valid = time[split_time:]
x_valid = series[split_time:]

naive_forecast = series[split_time - 1: -1]
plt.figure()
plot_series(time_valid, x_valid, label="Series")
plot_series(time_valid, naive_forecast, label="Forcast")

plt.figure()
plot_series(time_valid, x_valid, start=0, end=150, label="Series")
plot_series(time_valid, naive_forecast, start=1, end=151, label="Forecast")

errors = naive_forecast - x_valid
abs_errors = np.abs(errors)
mean = abs_errors.mean()
print(mean, "mean error")
plt.show()
