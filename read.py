import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

def find_max_in_each_line(file_path):
    max_values = []
    with open(file_path, 'r') as file:
        for line in file:
            # Remove brackets and whitespace, then split by comma
            numbers = [float(num) for num in line.strip()[1:-1].split(',')]
            max_values.append(sorted(numbers))
    return max_values

# Example usage:
file_path = 'scratch.txt'  # Replace with your actual file path
data_lists = find_max_in_each_line(file_path)


# # Create one subplot per list
# fig, axes = plt.subplots(len(data_lists), 1, figsize=(6, 4 * len(data_lists)))

for i, data in enumerate(data_lists):
    data_min, data_max = min(data), max(data)
    
    # Create bins for histogram
    bins = np.linspace(data_min, data_max, 50)  # More bins for smoother histogram
    
    # Plot histogram (normalized to density)
    plt.figure(figsize=(6,4))
    plt.hist(data, bins=bins, density=True, alpha=0.4, edgecolor='black', label='Histogram')
    
    # Compute KDE for smoothing
    kde = gaussian_kde(data)
    x_vals = np.linspace(data_min, data_max, 1000)
    kde_vals = kde(x_vals)
    
    plt.plot(x_vals, kde_vals, color='red', label='KDE Smooth')
    plt.title(f'AMD XT 7900 Throughput Space per Input Size (elements=2^22)')
    plt.xlabel('Throughput GB/s')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True)
    
    filename = f'smooth_bucket_plot_max_{data_max:.2f}.png'
    plt.savefig(filename)
    plt.close()
    print(f'Saved {filename}')