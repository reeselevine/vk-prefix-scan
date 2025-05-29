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

def freedman_diaconis_bins(data):
    q25, q75 = np.percentile(data, [25, 75])
    iqr = q75 - q25
    bin_width = 2 * iqr / len(data)**(1/3)
    if bin_width == 0:  # Avoid divide by zero
        bin_width = (max(data) - min(data)) / 10
    return max(1, int(np.ceil((max(data) - min(data)) / bin_width)))

def shimazaki_shinomoto_bins(data, k_min=4, k_max=100):
    data = np.asarray(data)
    N = len(data)
    D = np.max(data) - np.min(data)
    
    best_cost = float('inf')
    best_bins = None

    for k in range(k_min, k_max + 1):
        h = D / k
        edges = np.linspace(np.min(data), np.max(data), k + 1)
        counts, _ = np.histogram(data, bins=edges)
        mean = np.mean(counts)
        variance = np.var(counts)
        cost = (2 * mean - variance) / (h**2)

        if cost < best_cost:
            best_cost = cost
            best_bins = k

    return best_bins

size_ = 14
for i, data in enumerate(data_lists):
    data_min, data_max = min(data), max(data)
    
    # Create bins for histogram
    # bins = np.linspace(data_min, data_max, 50)  # More bins for smoother histogram
    # num_bins = freedman_diaconis_bins(data)
    # bins = np.linspace(data_min, data_max, num_bins + 1)

    bins = shimazaki_shinomoto_bins(data)
    # Plot histogram (normalized to density)
    plt.figure(figsize=(6,4))
    plt.hist(data, bins=bins, density=False, alpha=0.4, edgecolor='black', label='Histogram')
    
    # Compute KDE for smoothing
    #kde = gaussian_kde(data)
    #x_vals = np.linspace(data_min, data_max, 1000)
    #kde_vals = kde(x_vals)
    
    #plt.plot(x_vals, kde_vals, color='red', label='KDE Smooth')
    plt.title(f'Nvidia RTX 4070 @ 2^{size_} elements')
    plt.xlabel('Throughput GB/s')
    plt.ylabel(f'Configurations (total={len(data)})')
    plt.legend()
    plt.grid(True)
    
    filename = f'smooth_bucket_plot_max_{data_max:.2f}.png'
    plt.savefig(filename)
    plt.close()
    print(f'Saved {filename}')
    size_ += 1 