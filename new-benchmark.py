import os
import subprocess
import re
import statistics

# Path to the executable
executable = "./build/prefix-scan.run"

# Initial values
device = 1
workgroups = 128  # Initial -w value
threads = 64      # Initial -t value
p = 1
b = 'a'
n = 100  # Number of runs per configuration

# Set the maximum limits for workgroups and threads
max_workgroups = 2048
max_threads = 1024

# Regex patterns to extract throughput and error
throughput_pattern = re.compile(r'Throughput:\s*(\d+(\.\d+)?)')
error_pattern = re.compile(r'debug: (1|0)')

# Dictionary to store throughput analysis for each w * t
analysis = {}

# Function to execute the command and collect output
def run_command(command):
    result = subprocess.run(command, shell=True, stdout=subprocess.PIPE, text=True)
    return result.stdout

# Function to calculate statistics
def calculate_statistics(output):
    throughput_data = []
    error_data = []
    
    for line in output.splitlines():
        match = throughput_pattern.search(line)
        error_match = error_pattern.search(line)
        if match:
            throughput = float(match.group(1))
            throughput_data.append(throughput)
        if error_match:
            error = int(error_match.group(1))
            error_data.append(error)
    
    if throughput_data:
        avg_throughput = statistics.mean(throughput_data)
        var_throughput = statistics.variance(throughput_data)
    else:
        avg_throughput = var_throughput = 0.0

    if error_data:
        error_rate = 1 - sum(error_data) / len(error_data)
    else:
        error_rate = 0.0

    return avg_throughput, var_throughput, error_rate

# Function to update analysis based on w * t
def update_analysis(w, t, avg_throughput, var_throughput, error_rate):
    w_times_t = w * t
    if w_times_t not in analysis:
        analysis[w_times_t] = []
    
    analysis[w_times_t].append({
        'w': w,
        't': t,
        'avg_throughput': avg_throughput,
        'var_throughput': var_throughput,
        'error_rate': error_rate
    })

# Benchmark loop
done = False
while workgroups <= max_workgroups and threads <= max_threads and done == False:
    if workgroups == max_workgroups and threads == max_threads:
        done = True 
    # Run the program n times and collect the output
    output = ""
    for _ in range(n):
        command = f"{executable} -d {device} -w {workgroups} -t {threads} -p {p} -b '{b}'"
        output += run_command(command)

    # Calculate statistics from the output
    avg_throughput, var_throughput, error_rate = calculate_statistics(output)
    
    # Update the analysis dictionary based on w * t
    update_analysis(workgroups, threads, avg_throughput, var_throughput, error_rate)
    
    # Print current configuration statistics
    print(f"Configuration -w {workgroups} -t {threads}:")
    print(f"  Average Throughput: {avg_throughput}")
    print(f"  Variance of Throughput: {var_throughput}")
    print(f"  Error Rate: {error_rate * 100:.2f}%")
    
    # Alternate increasing workgroups and threads by powers of 2
    if threads < max_threads:
        threads *= 2
    else:
        threads = 64  # Reset threads to initial value    
        if workgroups < max_workgroups:
            workgroups *= 2



# Analyze results: find highest throughput for each w * t group
print("\nSummary of results (grouped by w * t):")
for w_times_t, data_list in analysis.items():
    # Find the combination with the highest throughput for each w * t
    best_combination = max(data_list, key=lambda x: x['avg_throughput'])
    print(f"w * t = {w_times_t}:")
    print(f"  Best combination -w {best_combination['w']} -t {best_combination['t']}:")
    print(f"    Average Throughput: {best_combination['avg_throughput']}")
    print(f"    Variance of Throughput: {best_combination['var_throughput']}")
    print(f"    Error Rate: {best_combination['error_rate'] * 100:.2f}%")