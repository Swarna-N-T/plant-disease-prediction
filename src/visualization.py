import matplotlib.pyplot as plt
import numpy as np

def plot_histograms(X_features):
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    avg_features = X_features.mean(axis=0)
    plt.bar(range(len(avg_features)), avg_features)
    plt.title('Average Feature Histogram (All Images)')
    plt.xlabel('Pixel Intensity (0-255)')
    plt.ylabel('Average Pixel Count')

    plt.subplot(1, 2, 2)
    for i in range(min(5, len(X_features))):
        plt.plot(X_features[i], alpha=0.7, label=f'Image {i+1}')
    plt.title('Individual Feature Histograms (First 5 Images)')
    plt.xlabel('Pixel Intensity (0-255)')
    plt.ylabel('Pixel Count')
    plt.legend()
    plt.tight_layout()
    plt.show()
