import os
import subprocess
import re
import statistics
import matplotlib.pyplot as plt
import math
import pandas as pd
import time
start = time.time()

fig, ax1 = plt.subplots(1, 1, figsize=(10, 6))

error_commands = [('Begin:', None)]
# Set the maximum limits for threads
max_threads = 1024
max_size = 1073741824
#max_size = 65536

# Regex patterns to extract throughput and error
throughput_pattern = re.compile(r'Throughput:\s*(\d+(\.\d+)?)')
error_pattern = re.compile(r'debug: (1|0)')

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
def update_analysis(w, t, b_option, bs, avg_throughput, var_throughput, error_rate, analysis):
    w_times_t_times_bs = w * t * bs
    if w_times_t_times_bs not in analysis:
        analysis[w_times_t_times_bs] = []
    analysis[w_times_t_times_bs].append({
        'w': w,
        't': t,
        'avg_throughput': avg_throughput,
        'var_throughput': var_throughput,
        'error_rate': error_rate,
        'b_option': b_option,
        'bs': bs
    })


# Function to update analysis based on w * t and 'b' option
def update_blit_analysis(w, t, bs, avg_throughput, var_throughput, error_rate, analysis):
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


def run_benchmark(_blit, _max_size, _device, _color, _executable, _n, _p, _max_batch_size, _min_threads, _label, _power_of_two):
    analysis = {}
    
    if not _blit:
        # Benchmark loop
        power_of_two = _power_of_two
        for ll in range(1, _max_batch_size):
            workgroups = 32  # Initial -w value
            threads = _min_threads      # Initial -t value
            while (workgroups * threads <= _max_size):
                # Test each '-b' option
                if not (power_of_two * workgroups * threads >= _max_size):
                    output = ""
                    for b_option in ['a', 'c']:
                        # Run the program n times and collect the output
                        alt = 1
                        for _ in range(_n):
                            command = f"{_executable} -d {_device} -w {workgroups} -t {threads} -p {_p} -b '{b_option}' -a '{alt}' -s'{power_of_two}'"
                            output += run_command(command)
                            alt += 1

                        # Calculate statistics from the output
                        avg_throughput, var_throughput, error_rate = calculate_statistics(output)
                    
                        # Update the analysis dictionary based on w * t and 'b' option
                        update_analysis(workgroups, threads, b_option, power_of_two, avg_throughput, var_throughput, error_rate, analysis)
                    
                        # Print current configuration statistics
                        print(f"Configuration -w {workgroups} -t {threads} -b '{b_option}', -bs {power_of_two}:")
                        print(f"  Average Throughput: {avg_throughput}")
                        print(f"  Variance of Throughput: {var_throughput}")
                        print(f"  Error Rate: {error_rate * 100:.2f}%")
                        if (error_rate > .001):
                            error_commands.append((error_rate, f"{_executable} -d {_device} -w {workgroups} -t {threads} -p {_p} -b '{b_option}' -a '{alt}' -s'{power_of_two}'"))
                # Alternate increasing workgroups and threads by powers of 2
                if threads < max_threads:
                    threads *= 2
                else:
                    threads = _min_threads  # Reset threads to initial value
                    workgroups *= 2
            power_of_two = power_of_two << 1
        # Analyze results: find highest throughput for each w * t group
        max_throughput_results = []
        print("\nSummary of results (grouped by w * t and -b option):")
        for w_times_t_bs, b_data in analysis.items():
            # Find the best combination of w * t and 'b' option with highest throughput
            best_data = max(b_data, key=lambda item: item['avg_throughput'])
            print(f"w * t * bs = {w_times_t_bs}, best -b option = '{best_data['b_option']}':")
            print(f"  Best combination -w {best_data['w']} -t {best_data['t']} -s {best_data['bs']} -b '{best_data['b_option']}':")
            print(f"    Average Throughput: {best_data['avg_throughput']}")
            print(f"    Variance of Throughput: {best_data['var_throughput']}")
            print(f"    Error Rate: {best_data['error_rate'] * 100:.2f}%")
            

            # Store max throughput for plotting
            max_throughput_results.append((w_times_t_bs, round(best_data['avg_throughput'], 2), 'raking method' if best_data['b_option'] == 'a' else 'blelloch method', round(math.sqrt(best_data['var_throughput']), 2), best_data['w'], best_data['t'], best_data['bs']))

        # Sort results by w * t for better plotting
        max_throughput_results.sort(key=lambda x: x[0])

        # Plot the results
        w_times_t_bs_values = [x[0] for x in max_throughput_results]
        throughput_values = [x[1] for x in max_throughput_results]
        b_options = [x[2] for x in max_throughput_results]  # Track b option for labeling
        std_values = [x[3] for x in max_throughput_results]
        w_t_bs_values = [(x[4], x[5], x[6]) for x in max_throughput_results]


        ax1.errorbar(w_times_t_bs_values, throughput_values, yerr=std_values, capsize=5, marker='o', linestyle='-', color=_color, label=_label)
        # for i, b in enumerate(b_options):
        #     ax1.annotate(f"b='{b}'\n", (w_times_t_bs_values[i], throughput_values[i]), textcoords="offset points", xytext=(0, 30), ha='center', fontsize=7)
        # for i, wt in enumerate(w_t_bs_values):
        #     ax1.annotate(f"w='{wt[0]}'\nt='{wt[1]}'\nbs='{wt[2]}", (w_times_t_bs_values[i], throughput_values[i]), textcoords="offset points", xytext=(0, 9), ha='center', fontsize=7)

        df = pd.DataFrame()
        df['BYTES'] = ['GB/s', 'LOCAL METHOD', 'STD DEV', 'WORKGROUPS', 'THREADS', 'BATCH SIZE']
        for el in max_throughput_results:
            temp = [i for i in el]
            del temp[0]
            df[el[0]] = temp 

        fig3, ax3 = plt.subplots(figsize=(10, 6))  # Adjust size for the table
        ax3.axis('off')  # Turn off the axes for the table
        table_data = [list(df.columns)] + df.values.tolist()
        table = ax3.table(cellText=table_data, colLabels=None, cellLoc='center', loc='center')
        table.scale(1, 1)  # Scale the table to make it readable
        fig3.savefig('pd-folder/' + _label + '.png',dpi=300)
    else:
        # Benchmark loop
        power_of_two = _power_of_two
        for ll in range(1, _max_batch_size):
            workgroups = 32  # Initial -w value
            threads = _min_threads      # Initial -t value
            while (workgroups * threads <= _max_size):
                # Test each '-b' option
                if not (power_of_two * workgroups * threads >= _max_size):
                    output = ""
                    alt = 1
                    for _ in range(_n):
                        command = f"{_executable} -d {_device} -w {workgroups} -t {threads} -a '{alt}' -s'{power_of_two}'"
                        output += run_command(command)
                        alt += 1

                    # Calculate statistics from the output
                    avg_throughput, var_throughput, error_rate = calculate_statistics(output)
                
                    # Update the analysis dictionary based on w * t and 'b' option
                    update_blit_analysis(workgroups, threads, power_of_two, avg_throughput, var_throughput, error_rate, analysis)
                
                    # Print current configuration statistics
                    print(f"Configuration -w {workgroups} -t {threads}, -bs {power_of_two}:")
                    print(f"  Average Throughput: {avg_throughput}")
                    print(f"  Variance of Throughput: {var_throughput}")
                    print(f"  Error Rate: {error_rate * 100:.2f}%")
                    if (error_rate > .001):
                            error_commands.append((error_rate, f"{_executable} -d {_device} -w {workgroups} -t {threads} -a '{alt}' -s'{power_of_two}'"))
                # Alternate increasing workgroups and threads by powers of 2
                if threads < max_threads:
                    threads *= 2
                else:
                    threads = _min_threads  # Reset threads to initial value
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
            max_throughput_results.append((w_times_t_bs, round(best_data['avg_throughput'], 2), round(math.sqrt(best_data['var_throughput']), 2), best_data['w'], best_data['t'], best_data['bs']))

        # Sort results by w * t for better plotting
        max_throughput_results.sort(key=lambda x: x[0])

        # Plot the results
        w_times_t_bs_values = [x[0] for x in max_throughput_results]
        throughput_values = [x[1] for x in max_throughput_results]
        std_values = [x[2] for x in max_throughput_results]
        w_t_bs_values = [(x[3], x[4], x[5]) for x in max_throughput_results]


        ax1.errorbar(w_times_t_bs_values, throughput_values, yerr=std_values, capsize=5, marker='o', linestyle='-', color=_color, label=_label)
        # for i, wt in enumerate(w_t_bs_values):
        #     ax1.annotate(f"w='{wt[0]}'\nt='{wt[1]}'\nbs='{wt[2]}", (w_times_t_bs_values[i], throughput_values[i]), textcoords="offset points", xytext=(0, 9), ha='center', fontsize=7)

        df = pd.DataFrame()
        df['BYTES'] = ['GB/s', 'STD DEV', 'WORKGROUPS', 'THREADS', 'BATCH SIZE']
        for el in max_throughput_results:
            temp = [str(i) for i in el]
            del temp[0]
            df[el[0]] = temp
            #df[el[0]] = df[el[0]].astype(str)

        fig4, ax4 = plt.subplots(figsize=(10, 6))  # Adjust size for the table
        ax4.axis('off')  # Turn off the axes for the table
        table_data = [list(df.columns)] + df.values.tolist()
        table = ax4.table(cellText=table_data, colLabels=None, cellLoc='center', loc='center')
        table.scale(1, 1)  # Scale the table to make it readable
        fig4.savefig('pd-folder/' + _label + '.png',dpi=300)

        
            
#run_benchmark(False, max_size, 0, 'blue', "./main-bench-path/bp-prefix-scan.run", 5,  1, 5, 64, "AMD XT 7900 - bp & par")
run_benchmark(False, max_size, 0, 'blue', "./main-bench-path/bp-prefix-scan.run", 5,  1, 2, 64, "AMD XT 7900 - bp & par", 8)
run_benchmark(False, max_size, 0, 'green', "./main-bench-path/prefix-scan.run", 5, 1, 5, 64, "AMD XT 7900 - no bp & par", 8)
# run_benchmark(True, max_size, 0, 'black', "./main-bench-path/blit.run", 5, 1, 2, 64, "AMD XT 7900 - blit", 1)
#run_benchmark(False, max_size, 0, 'orange', "./main-bench-path/bp-prefix-scan.run", 5, 0, 5, 64, "AMD XT 7900 - bp & no par", 1)
#run_benchmark(False, max_size, 0, 'red', "./main-bench-path/prefix-scan.run", 5, 0, 5, 64, "AMD XT 7900 - no bp & no par", 1)


# Plot settings
fig.suptitle("Best throughput parameterized on w, t and reduction type")
ax1.set_xlabel("workgroups * threads * batch_size")
ax1.set_ylabel("Throughput")
ax1.grid(False)
ax1.set_xscale('log', base=2)
# Configure and display legend without error bars
handles, labels = ax1.get_legend_handles_labels()
handles = [h[0] for h in handles]
ax1.legend(handles, labels, loc='upper left', numpoints=1)

fig.savefig("plots/exhaustive-bench-complete.png")

for i in error_commands:
    print(i)    

end = time.time()

print(end - start)
print("seconds")
# save data so we cna see if different runs re marginlly different or very different.
# do this by graphing a scatter plot so you can see all of the  points compared to eahcother 

