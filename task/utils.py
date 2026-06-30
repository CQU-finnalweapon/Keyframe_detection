import numpy as np
import matplotlib.pyplot as plt

def _smooth(data, window_size=5, visualize=False):
    if window_size % 2 == 0:
        print("window size must be odd, add 1 autonomously.")
        window_size += 1
    data = np.pad(data, (window_size // 2, window_size // 2), mode='edge')
    data_smoothed = np.convolve(data, np.ones(window_size) / window_size, mode='valid')
    
    ## plot the oritinal and smoothed data
    if visualize == True:
        plt.figure(figsize=(10, 5))
        plt.plot(data, label='Original Data', alpha=0.5)
        plt.plot(data_smoothed, label='Smoothed Data', color='red')
        plt.title('Data Smoothing')
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.legend()
        plt.show()
        plt.savefig("data_smoothing.png")
        plt.close()
        print(f"Data smoothed with window size {window_size}.")

    if len(data_smoothed) == 0:
        print("Warning: smoothed data is empty, check the input data.")
        return data
    
    return data_smoothed