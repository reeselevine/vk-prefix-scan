import matplotlib.pyplot as plt
import numpy as np

def parse_data(file_path):
    # Dictionary to store data sets with their throughput values
    data_sets = {}
    current_set_name = None
    current_data = []

    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            
            # Check if the line is a dataset name (it's not a data line)
            if not line.startswith("{") and not line.startswith("}") and len(line) > 0:
                # If there's existing data, save the previous dataset
                if current_set_name and current_data:
                    data_sets[current_set_name] = current_data
                current_set_name = line.strip()  # Set the name of the current set
                current_data = []  # Reset the data for the new set
            
            elif line.startswith("{") and 'throughput' in line:
                # Extract the throughput value from the line
                try:
                    throughput = float(line.split('throughput')[1].split(':')[1].split(',')[0].strip())
                    current_data.append(throughput)
                except ValueError:
                    pass  # Skip lines that don't contain valid throughput values

        # Don't forget to add the last dataset
        if current_set_name and current_data:
            data_sets[current_set_name] = current_data

    return data_sets

def plot_throughput(data_sets, save_path):
    # Find the maximum number of data points across all sets
    max_data_points = max(len(data_sets[key]) for key in data_sets)

    # Generate input size values starting from 2^10, 2^11, ... to accommodate the largest dataset
    input_sizes = [2**(10 + i) for i in range(max_data_points)]

    # Plotting the data
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot each dataset's throughput against the input size
    for label, data in data_sets.items():
        # Ensure each set has the same number of data points by truncating the excess data points
        if len(data) < max_data_points:
            data += [data[-1]] * (max_data_points - len(data))  # Fill in with the last throughput value

        # Plot the data for each dataset as a separate line with a label for the legend
        ax1.plot(input_sizes, data, label=label)

    ax1.set_xlabel('Input Size (2^x)')
    ax1.set_ylabel('Throughput')
    ax1.set_title('Throughput vs. Input Size')

    # Set the x-axis to be logarithmic with base 2
    ax1.set_xscale('log', base=2)

    # Show the legend
    ax1.legend()

    ax1.grid(True, which="both", ls="--")

    # Save the plot to a file (PNG format) without showing it
    plt.savefig(save_path, format='png')

# File path to the input data
file_path = 'parse-file.txt'  # Replace with the actual path to your data file

# Path where the plot will be saved
save_path = 'throughput_plot.png'

# Parse the data and plot it, saving the plot to a file
data_sets = parse_data(file_path)
plot_throughput(data_sets, save_path)

print(f"Plot saved to {save_path}")