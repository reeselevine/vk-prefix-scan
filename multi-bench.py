
import os
import subprocess
import re
import statistics
import matplotlib.pyplot as plt
import math



# Path to the executable
executable = "./build/prefix-scan.run"
devices = ["Intel ARC A770", "Nvidia GTX 4070", "Intel UHD"]
colors = ["r", "g", "b"]
max_workgroups_numbers = [1024, 2048, 4096]
file_names = ['all-w-dif-reductions2^23.png', 'all-w-dif-reductions2^24.png', 'all-w-dif-reductions2^25.png']
for ii in range (0, 3):
    fig, ax1 = plt.subplots(1, 1, figsize=(10, 6))
    for jj in range(0, 3):
        # Initial values
        device = jj
        workgroups = 128  # Initial -w value
        threads = 32      # Initial -t value
        p = 1
        n = 10   # Number of runs per configuration
        batch_size = 8
        device_name = devices[jj]
        color = colors[jj]
        # Set the maximum limits for workgroups and threads
        max_workgroups = max_workgroups_numbers[ii]
        max_threads = 1024
    

        # Regex patterns to extract throughput and error
        throughput_pattern = re.compile(r'Throughput:\s*(\d+(\.\d+)?)')
        error_pattern = re.compile(r'debug: (1|0)')

        # Dictionary to store throughput analysis for each w * t and 'b' option
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

        # Function to update analysis based on w * t and 'b' option
        def update_analysis(w, t, b_option, avg_throughput, var_throughput, error_rate):
            w_times_t = w * t * batch_size
            if w_times_t not in analysis:
                analysis[w_times_t] = []
            analysis[w_times_t].append({
                'w': w,
                't': t,
                'avg_throughput': avg_throughput,
                'var_throughput': var_throughput,
                'error_rate': error_rate,
                'b_option': b_option
            })

        # Benchmark loop
        done = False
        while workgroups <= max_workgroups and threads <= max_threads and done == False:
            if workgroups == max_workgroups and threads == max_threads:
                done = True 
            # Test each '-b' option
            for b_option in ['a', 'c']:
                # Run the program n times and collect the output
                output = ""
                for _ in range(n):
                    command = f"{executable} -d {device} -w {workgroups} -t {threads} -p {p} -b '{b_option}'"
                    output += run_command(command)

                # Calculate statistics from the output
                avg_throughput, var_throughput, error_rate = calculate_statistics(output)
                
                # Update the analysis dictionary based on w * t and 'b' option
                update_analysis(workgroups, threads, b_option, avg_throughput, var_throughput, error_rate)
                
                # Print current configuration statistics
                print(f"Configuration -w {workgroups} -t {threads} -b '{b_option}':")
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
        max_throughput_results = []
        print("\nSummary of results (grouped by w * t and -b option):")
        for w_times_t, b_data in analysis.items():
            # Find the best combination of w * t and 'b' option with highest throughput
            best_data = max(b_data, key=lambda item: item['avg_throughput'])
            print(f"w * t = {w_times_t}, best -b option = '{best_data['b_option']}':")
            print(f"  Best combination -w {best_data['w']} -t {best_data['t']} -b '{best_data['b_option']}':")
            print(f"    Average Throughput: {best_data['avg_throughput']}")
            print(f"    Variance of Throughput: {best_data['var_throughput']}")
            print(f"    Error Rate: {best_data['error_rate'] * 100:.2f}%")

            # Store max throughput for plotting
            max_throughput_results.append((w_times_t, best_data['avg_throughput'], best_data['b_option'], math.sqrt(best_data['var_throughput']), best_data['w'], best_data['t']))

        # Sort results by w * t for better plotting
        max_throughput_results.sort(key=lambda x: x[0])

        # Plot the results
        w_times_t_values = [x[0] for x in max_throughput_results]
        throughput_values = [x[1] for x in max_throughput_results]
        b_options = [x[2] for x in max_throughput_results]  # Track b option for labeling
        std_values = [x[3] for x in max_throughput_results]
        w_t_values = [(x[4], x[5]) for x in max_throughput_results]

        
        ax1.errorbar(w_times_t_values, throughput_values, yerr=std_values, capsize=5, marker='o', linestyle='-', color=color, label=device_name)
        for i, b in enumerate(b_options):
            ax1.annotate(f"b='{b}'\n", (w_times_t_values[i], throughput_values[i]), textcoords="offset points", xytext=(0, 20), ha='center')
        for i, wt in enumerate(w_t_values):
            ax1.annotate(f"w='{wt[0]}'\nt='{wt[1]}'", (w_times_t_values[i], throughput_values[i]), textcoords="offset points", xytext=(0, 10), ha='center')


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

    fig.savefig("hello.png")