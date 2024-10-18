import re
import sys
import statistics

def sum_throughput(file_path):
    total_throughput = 0.0
    total_error = 0.0
    throughput_pattern = re.compile(r'Throughput:\s*(\d+(\.\d+)?)')
    error_pattern = re.compile(r'debug: (1|0)')
    throughput_data = list()
    error_data = list()

    try:
        with open(file_path, 'r') as file:
            cnt = 0
            for line in file:
                match = throughput_pattern.search(line)
                error_match = error_pattern.search(line)
                if match:
                    throughput = float(match.group(1))
                    throughput_data.append(throughput)
                if error_match:
                    error = int(error_match.group(1))
                    error_data.append(error)

            var_throughput = statistics.variance(throughput_data)
            total_throughput = statistics.mean(throughput_data)
            total_error = 1 - sum(error_data) / len(error_data)

        print(f"Average Throughput: {total_throughput}")
        print(f"Throughput Variance: {var_throughput}")
        print(f"Percent Error: {total_error}")
    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

# Example usage
if __name__ == "__main__":
    sum_throughput('throughput.dat')
