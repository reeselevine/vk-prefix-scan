import os
import subprocess
import re
import statistics
import matplotlib.pyplot as plt
import math
#import pandas as pd
import time
#start = time.time()

error_commands = [('Begin:', None)]
fig, ax1 = plt.subplots(1, 1, figsize=(10, 6))
#fig, ax2 = plt.subplots(figsize=(10, 6))
# Regex patterns to extract throughput and error
throughput_pattern = re.compile(r'Throughput:\s*(\d+(\.\d+)?)')
error_pattern = re.compile(r'debug: (1|0)')



file_name = "batch-vec-loads-no-branch"
plot_name = "Vulkan prefix-sum vec loads"
x_label = "workgroups * threads * batch_size"
y_label = "Throughput"
min_size = 12
max_size = 26
min_bs = 0
max_bs = 2
min_threads = 5
max_threads = 10
min_workgroups = 5
max_workgroups = max_size - min_threads
throughputs_ = [list() for _ in range(10, max_size - 1)]



def run_command(command):
    #print(command)
    result = subprocess.run(command, shell=True, stdout=subprocess.PIPE, text=True)
    #print(result)
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
            print(throughput)
        if error_match:
            error = int(error_match.group(1))
            error_data.append(error)

    if throughput_data:
        if len(throughput_data) != 1:
            avg_throughput = statistics.mean(throughput_data)
            var_throughput = statistics.variance(throughput_data)
        else:
            avg_throughput = throughput_data[0]
            var_throughput = 0
    else:
        avg_throughput = var_throughput = 0.0

    if error_data:
        error_rate = 1 - sum(error_data) / len(error_data)
    else:
        error_rate = 0.0

    return avg_throughput, var_throughput, error_rate


def main():
    run(_blit=False, _device=1, _executable="build/prefix-scan.run", _n=2, _label="Nvidia GTX 4070 any lookback vec loads", _color="red", _warmup=1)
    #run(_blit=True, _device=1, _executable="build/blit.run", _n=2, _p=0, _label="Nvidia GTX 4070 blit", _color="orange", _warmup=2)
    #run(_blit=True, _device=1, _executable="build/blit.run", _n=16, _p=0, _label="blit", _color="blue", _warmup=5)

def run(_blit, _device, _executable, _n, _label, _color, _warmup):
    # dict where keys r powers of 2 and values are dicts
    best_combinations = [dict() for _ in range(10, max_size - 1)]
    input_size = min_size
    last_max = 0
    sleeps = False
    while input_size <= max_size:
        max_throughput = 0
        max_var_throughput = 0
        max_error_rate = 0
        max_command = ""
        # start # - input over all params
        print("2 ^ " + str(input_size))
        #for alg in ['a', 'c']:
        for alg in ['a']:
            for _p in [0, 1]:
                for bs in (2**p for p in range(min_bs, max_bs + 1)):
                    for t in (2**p for p in range(min_threads, max_threads + 1)):
                        for w in (2**p for p in range(min_workgroups, max_workgroups + 1)):
                            # end # - input over all params
                            if bs * t * w == 2 ** input_size:
                                alt = 1
                                output = ""
                                warmup_output = ""

                                for i in range(_warmup):
                                    if not _blit:
                                        command = f"{_executable} -d {_device} -w {w} -t {t} -p {_p} -b '{alg}' -a '{alt}' -s'{bs}'"
                                        print("Warmup", i, f": -w {w} -t {t} -s '{bs}' -b '{alg}' ")
                                    else:
                                        command = f"{_executable} -d {_device} -w {w} -t {t} -a '{alt}' -s'{bs}'"
                                        print("Warmup", i, f": -w {w} -t {t} -s '{bs}' ")
                                    warmup_output += run_command(command)
                                    alt += 1
                            
                                alt = 1

                                for i in range(_n):
                                    if not _blit:
                                        command = f"{_executable} -d {_device} -w {w} -t {t} -p {_p} -b '{alg}' -a '{alt}' -s'{bs}'"
                                        print(f"-w {w} -t {t} -s '{bs}' -b '{alg}' ")
                                    else:
                                        command = f"{_executable} -d {_device} -w {w} -t {t} -a '{alt}' -s'{bs}'"
                                        print(f"-w {w} -t {t} -s '{bs}' ")
                                    if sleeps == True:
                                        time.sleep(2)
                                    output += run_command(command)
                                    alt += 1

                                avg_throughput, var_throughput, error_rate = calculate_statistics(output)
                                if avg_throughput > max_throughput:
                                    max_throughput = avg_throughput
                                    max_var_throughput = var_throughput
                                    max_error_rate = error_rate
                                    max_command = command
                                if error_rate > 0:
                                    error_commands.append(command)
                                throughputs_[input_size - min_size].append(avg_throughput)
                                for i in throughputs_:
                                    print(i)

        best_combinations[input_size - min_size]["throughput"] = round(max_throughput)
        best_combinations[input_size - min_size]["var_throughput"] = round(max_var_throughput)
        best_combinations[input_size - min_size]["error_rate"] = max_error_rate
        best_combinations[input_size - min_size]["command"] = max_command
                            
        # if last_max > max_throughput:
        #     sleeps = True
        #     for _ in range(0, 10):
        #         print("WE GOING TO SLEEP BOY")
        # else:
        #     input_size = input_size + 1
        #     last_max = max_throughput
        #     sleeps = False
        input_size = input_size + 1
    input_sizes = [input_size for input_size in (2**p for p in range(min_size, max_size + 1))]
    throughputs = [input_size["throughput"] for input_size in best_combinations]
    std_values = [input_size["var_throughput"] for input_size in best_combinations]
    ax1.errorbar(input_sizes, throughputs, yerr=std_values, capsize=5, marker='o', linestyle='-', color=_color, label=_label)
    with open(file_name + '.txt', 'a') as output:
        output.write(_label + "\n\n")
        for entry in best_combinations:
            output.write(str(entry) + "\n")
        output.write("\n\n")


main()



# Create bar chart on ax2
#choice = 20
#ax2.bar([i for i in range(0, len(throughputs_[choice - min_size]))], throughputs_[choice - min_size], color='blue')

# # Add title and labels
# ax2.set_title("Sample Bar Graph")
# ax2.set_xlabel("Categories")
# ax2.set_ylabel("Values")

# # Save the plot instead of showing it
# plt.savefig("bar_graph.png")


ax1.set_xscale('log', base=2)
# Plot settings
fig.suptitle(plot_name)
ax1.set_xlabel(x_label)
ax1.set_ylabel(y_label)
ax1.grid(False)
ax1.set_xscale('log', base=2)
# Configure and display legend without error bars
handles, labels = ax1.get_legend_handles_labels()
handles = [h[0] for h in handles]
ax1.legend(handles, labels, loc='upper left', numpoints=1)

fig.savefig(file_name + '.png')


with open(file_name + '.txt', 'a') as output:
    output.write("\n\nerror inputs:\n\n" )
    for entry in error_commands:
        output.write(str(entry) + "\n")



                    