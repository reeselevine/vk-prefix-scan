import os
import subprocess
import re
import statistics
import matplotlib.pyplot as plt
import math
import pickle 

# Path to the executable
executable = "./build/blit.run"
fig, ax1 = plt.subplots(1, 1, figsize=(10, 6))

########
# Initial values
device = 0
min_threads = 64
threads = min_threads      # Initial -t value
p = 1
n = 5   # Number of runs per configuration
batch_size = 1
max_batch_size = 5
color = 'r'
# Set the maximum limits for  threads
max_threads = 1024
#######
max_size = 1073741824
#max_size = 65535

# Regex patterns to extract throughput and error
throughput_pattern = re.compile(r'Throughput:\s*(\d+(\.\d+)?)')
error_pattern = re.compile(r'debug: (1|0)')

# Dictionary to store throughput analysis for each w * t and 'b' option
analysis = {}

# Function to execute the command and collect output
def run_command(command):
    print(command)
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

# Function to update analysis based on w * t and 'b' option
def update_analysis(w, t, bs, avg_throughput, var_throughput, error_rate):
    w_times_t_times_bs = w * t * bs
    if w_times_t_times_bs not in analysis:
        analysis[w_times_t_times_bs] = []
    analysis[w_times_t_times_bs].append({
        'w': w,
        't': t,
        'avg_throughput': avg_throughput,
        'var_throughput': var_throughput,
        'error_rate': error_rate,
        'bs': bs
    })

# Benchmark loop
power_of_two = 1
for ll in range(1, max_batch_size):
    workgroups = 64  # Initial -w value
    threads = min_threads      # Initial -t value
    while (workgroups * threads <= max_size):
        # Test each '-b' option
        if not (power_of_two * workgroups * threads >= max_size):
            output = ""
            alt = 1
            for _ in range(n):
                command = f"{executable} -d {device} -w {workgroups} -t {threads} -a '{alt}' -s'{power_of_two}'"
                output += run_command(command)
                alt += 1

            # Calculate statistics from the output
            avg_throughput, var_throughput, error_rate = calculate_statistics(output)
        
            # Update the analysis dictionary based on w * t and 'b' option
            update_analysis(workgroups, threads, power_of_two, avg_throughput, var_throughput, error_rate)
        
            # Print current configuration statistics
            print(f"Configuration -w {workgroups} -t {threads}, -bs {power_of_two}:")
            print(f"  Average Throughput: {avg_throughput}")
            print(f"  Variance of Throughput: {var_throughput}")
            print(f"  Error Rate: {error_rate * 100:.2f}%")
        # Alternate increasing workgroups and threads by powers of 2
        if threads < max_threads:
            threads *= 2
        else:
            threads = min_threads  # Reset threads to initial value
            workgroups *= 2
    power_of_two = power_of_two << 1
# Analyze results: find highest throughput for each w * t group
max_throughput_results = []
print("\nSummary of results (grouped by w * t * bs):")
for w_times_t_bs, b_data in analysis.items():
    # Find the best combination of w * t and 'b' option with highest throughput
    best_data = max(b_data, key=lambda item: item['avg_throughput'])
    print(f"w * t * bs = {w_times_t_bs}:")
    print(f"  Best combination -w {best_data['w']} -t {best_data['t']} -s {best_data['bs']}:")
    print(f"    Average Throughput: {best_data['avg_throughput']}")
    print(f"    Variance of Throughput: {best_data['var_throughput']}")
    print(f"    Error Rate: {best_data['error_rate'] * 100:.2f}%")

    # Store max throughput for plotting
    max_throughput_results.append((w_times_t_bs, best_data['avg_throughput'], math.sqrt(best_data['var_throughput']), best_data['w'], best_data['t'], best_data['bs']))

# Sort results by w * t for better plotting
max_throughput_results.sort(key=lambda x: x[0])

# Plot the results
w_times_t_bs_values = [x[0] for x in max_throughput_results]
throughput_values = [x[1] for x in max_throughput_results]
std_values = [x[2] for x in max_throughput_results]
w_t_bs_values = [(x[3], x[4], x[5]) for x in max_throughput_results]


ax1.errorbar(w_times_t_bs_values, throughput_values, yerr=std_values, capsize=5, marker='o', linestyle='-', color=color, label="AMD XT 7900")
for i, wt in enumerate(w_t_bs_values):
    ax1.annotate(f"w='{wt[0]}'\nt='{wt[1]}'\nbs='{wt[2]}", (w_times_t_bs_values[i], throughput_values[i]), textcoords="offset points", xytext=(0, 9), ha='center', fontsize=7)


# Plot settings
fig.suptitle("Best throughput parameterized on w, t and bs")
ax1.set_xlabel("workgroups * threads * batch_size")
ax1.set_ylabel("Throughput")
ax1.grid(False)
ax1.set_xscale('log', base=2)
# Configure and display legend without error bars
handles, labels = ax1.get_legend_handles_labels()
handles = [h[0] for h in handles]
ax1.legend(handles, labels, loc='upper left', numpoints=1)

fig.savefig("exhaustive-bench-blit.png")

with open('exhaustive-bench-blit.pkl', 'wb') as f:
    pickle.dump(analysis, f)


# save data so we cna see if different runs re marginlly different or very different.
# unlock the workgroups