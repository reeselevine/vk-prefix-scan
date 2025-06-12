import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches
from collections import defaultdict
from functools import reduce



_file_name = "AllParams commands.txt"

output_file = _file_name + "_grouped"

with open(_file_name) as f:
        lines = f.readlines()



results = []
current_input_size = None

for line in lines:
    line = line.replace("'", "")
    # Check for input size group
    match_group = re.match(r"\((\d+)\)", line)
    if match_group:
        current_input_size = int(match_group.group(1))
        line = line[match_group.end():].strip()
    
    # Extract throughput and flags
    m = re.match(r"([0-9.]+)--build/prefix-scan.run (.*)", line)
    if m:
        throughput = float(m.group(1))
        flags = m.group(2)

        # Extract all relevant parameters
        args = dict(re.findall(r"-(\w) (\S+)", flags))
        results.append({
            "input_size": current_input_size,
            "throughput": throughput,
            "w": int(args.get('w', 0)),
            "t": int(args.get('t', 0)),
            "p": int(args.get('p', 0)),
            "b": args.get('b', ''),
            "a": args.get('a', ''),
            "s": int(args.get('s', 0)),
            "m": int(args.get('m', 0)),
        })

# Create DataFrame
df = pd.DataFrame(results)
#print(df)


# # Group by workgroup size
# grouped = df.groupby("input_size")



# # Mean throughput per input_size size
# print(grouped["throughput"].mean())



# grouped["throughput"].mean().plot(kind='bar')
# plt.title("Mean Throughput by Workgroup Size")
# plt.xlabel("Workgroup Size (-w)")
#plt.ylabel("Throughput")
# plt.savefig(output_file + ".png")


#Top 8 fixed configs
#############################

# #fixed params columns
# fixed_cols = ['p', 'b', 'a', 's', 'm']

# # 1. Average throughput by fixed params and input size (averaging over w, t)
# avg_throughput_by_fixed_input = (
#     df.groupby(['input_size'] + fixed_cols)['throughput']
#     .mean()
#     .reset_index()
# )

# # 2. Average over all input sizes to get top fixed param sets
# avg_throughput_by_fixed = (
#     avg_throughput_by_fixed_input.groupby(fixed_cols)['throughput']
#     .mean()
#     .reset_index()
#     .sort_values('throughput', ascending=False)
# )

# # 3. Select top 2 sets
# n = 1
# top_fixed_params = avg_throughput_by_fixed.iloc[:n]

# # Storage for both sets
# lines = []

# for i, (_, fixed_row) in enumerate(top_fixed_params.iterrows()):
#     # Get rows in df with matching fixed params
#     mask = pd.Series(True, index=df.index)
#     for col in fixed_cols:
#         mask &= df[col] == fixed_row[col]

#     filtered_df = df[mask]

#     #print(top_fixed_params[['throughput'] + fixed_cols])


#     # Find max throughput per input size (best w,t for each input size)
#     best_per_input = filtered_df.loc[
#         filtered_df.groupby('input_size')['throughput'].idxmax()
#     ][['input_size', 'throughput', 'w', 't']].sort_values('input_size')

#     label = f"Rank {i + 1}: " + ", ".join(f"{k}={fixed_row[k]}" for k in fixed_cols)
#     lines.append((best_per_input, label))
    

# # 4. Plot both
# plt.figure(figsize=(10,6))

# for line_data, label in lines:
#     label = label.replace("a=2, ", "")
#     plt.plot(line_data['input_size'], line_data['throughput'], marker='o', label=label)
#     print(line_data['throughput'])
#     for _, row in line_data.iterrows():
#         #plt.annotate(f"(w={row['w']},t={row['t']})",
#                     #  (row['input_size'], row['throughput']),
#                     #  textcoords="offset points",
#                     #  xytext=(0,10),
#                     #  ha='center',
#                     #  fontsize=8)
#         pass

# plt.xlabel("Input Size")
# plt.ylabel("Throughput")
# plt.title("Top 3 Fixed Param Sets: Throughput vs Input Size")
# plt.legend()
# plt.grid(True)

#############################

# # Step 1: Compute avg throughput by input_size and fixed params over all w,t
# avg_throughput_by_fixed_input = (
#     df.groupby(['input_size'] + fixed_cols)['throughput']
#     .mean()
#     .reset_index()
# )

# # Step 2: Compute avg throughput over input sizes by fixed params
# avg_throughput_by_fixed = (
#     avg_throughput_by_fixed_input.groupby(fixed_cols)['throughput']
#     .mean()
#     .reset_index()
# )

# # Step 3: Find best fixed params set (highest avg throughput)
# best_fixed_params = avg_throughput_by_fixed.sort_values('throughput', ascending=False).iloc[0]

# print("Best fixed parameters (p, b, a, s, m):")
# print(best_fixed_params)

# # Step 4: Filter original df to best fixed params rows
# mask = pd.Series(True, index=df.index)
# for col in fixed_cols:
#     mask &= df[col] == best_fixed_params[col]

# best_fixed_df = df[mask]

# # Step 5: For each input size, find max throughput and the (w, t) that produced it
# best_per_input = best_fixed_df.loc[
#     best_fixed_df.groupby('input_size')['throughput'].idxmax()
# ][['input_size', 'throughput', 'w', 't']].sort_values('input_size')

# # Step 6: Plot throughput vs input size and annotate (w,t)
# plt.figure(figsize=(10,6))
# plt.plot(best_per_input['input_size'], best_per_input['throughput'], marker='o', linestyle='-', linewidth=2)
# plt.xlabel('Input Size', fontsize=16)
# plt.ylabel('Throughput', fontsize=16)
# plt.title('Best Throughput vs Input Size for Best Device-wide Config', fontsize=16)

# print(best_per_input['throughput'])

# # Annotate each point with (w,t)
# for _, row in best_per_input.iterrows():
#     plt.annotate(f"(w={row['w']}, t={row['t']})",
#                  (row['input_size'], row['throughput']),
#                  textcoords="offset poinbest_six_combots",
#                  xytext=(0,10),
#                  ha='center',
#                  fontsize=8)

# plt.grid(True)


#Best avg reduction per workgroup size
##############################

# Find best-performing -b for each w
# Filter only b values of interest
# Filter to b ∈ {'a', 'c'}
# filtered = df[df['b'].isin(['a', 'c'])]

# # Get top 6 throughput values for each (w, b) pair
# top5_means = (
#     filtered
#     .sort_values('throughput', ascending=False)
#     .groupby(['w', 'b'])
#     .head(6)  # top 6 entries per (w, b)
#     .groupby(['w', 'b'])['throughput']
#     .mean()
#     .reset_index()
# )

# # Sort for consistent plotting
# top5_means = top5_means.sort_values('w')
# top5_means['w_label'] = top5_means['w'].apply(lambda x: f"$2^{{{int(np.log2(x))}}}$" if x > 0 and (x & (x-1)) == 0 else str(x))

# label_map = {
#     'a': 'SIMD Raking',
#     'c': 'Blelloch 1990'
# }
# top5_means['b_label'] = top5_means['b'].map(label_map)

# # Plot
# plt.figure(figsize=(10, 6))
# sns.barplot(data=top5_means, x='w_label', y='throughput', hue='b_label')
# plt.title('Top-6 Avg Throughput by Workgroup Size (-w) and Local Scan Type (-b)')
# plt.xlabel('Workgroup Size (-w)')
# plt.ylabel('Avg Throughput (Top 6 only)')
# plt.legend(title='Scan Algorithm')
# plt.grid(True, axis='y')
# plt.tight_layout()


#For each input_size, get top 3 configs by throughput
###############################3

#For each input_size, get top 3 configs by throughput
# top3_per_input = (
#     df
#     .sort_values(['input_size', 'throughput'], ascending=[True, False])
#     .groupby('input_size')
#     .head(8)
#     .copy()
# )

# # Assign rank per input size (1=best)
# top3_per_input['rank'] = top3_per_input.groupby('input_size')['throughput'].rank(ascending=False, method='first')

# # Pivot so each rank is a separate line, index=input_size, columns=rank, values=throughput
# pivot = top3_per_input.pivot(index='input_size', columns='rank', values='throughput')

# plt.figure(figsize=(12, 7))

# # Plot lines for rank 1, 2, 3
# for rank in [1, 2, 3]:
#     if rank in pivot.columns:
#         plt.plot(pivot.index, pivot[rank], marker='o', label=f'Top {int(rank)} config')

# # Annotate each input size with all three configs' summaries
# for input_size in pivot.index:
#     configs = top3_per_input[top3_per_input['input_size'] == input_size]
#     # Build annotation string for all three configs at this input size
#     annotations = []
#     for _, row in configs.iterrows():
#         # Customize this summary as you like, here showing w, t, b
#         #annotations.append(f"(w={row['w']}, t={row['t']}, b={row['b']})")
#         annotations.append(f"b={row['b']})")
    
#     annotation_text = "\n".join(annotations)
#     plt.annotate(
#         annotation_text,
#         (input_size, pivot.loc[input_size, 1]),
#         textcoords="offset points",
#         xytext=(5,5),
#         ha='left',
#         fontsize=8,
#         bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.3)
#     )

# plt.title('Top 3 Configurations Throughput vs Input Size')
# plt.xlabel('Input Size')
# plt.ylabel('Throughput')
# plt.legend()
# plt.grid(True)
# plt.tight_layout()

###############
# n = 1
# #For each input_size, get top 3 configs by throughput
# top3_per_input = (
#     df
#     .sort_values(['input_size', 'throughput'], ascending=[True, False])
#     .groupby('input_size')
#     .head(n)
#     .copy()
# )

# # Assign rank per input size (1=best)
# top3_per_input['rank'] = top3_per_input.groupby('input_size')['throughput'].rank(ascending=False, method='first')

# # Pivot so each rank is a separate line, index=input_size, columns=rank, values=throughput
# pivot = top3_per_input.pivot(index='input_size', columns='rank', values='throughput')

# print(pivot)

# plt.figure(figsize=(12, 7))

# # Plot lines for rank 1 -> n
# for rank in range(1, n + 1):
#     if rank in pivot.columns:
#         plt.plot(pivot.index, pivot[rank], marker='o', label=f'Top {int(rank)} config')

# # Annotate each input size with all three configs' summaries
# for input_size in pivot.index:
#     configs = top3_per_input[top3_per_input['input_size'] == input_size]
#     # Build annotation string for all three configs at this input size
#     annotations = []
#     m = 0
#     count = 0
#     for _, row in configs.iterrows():
#         # Customize this summary as you like, here showing w, t, b
#         #annotations.append(f"(w={row['w']}, t={row['t']}, b={row['b']})")
#         #annotations.append(f"b={row['b']})")
#         if row['b'] == "a":
#             annotations.append(f"Raking")
#         else:
#             annotations.append(f"Blelloch")
#             m += 1

#         count += 1
#     percentage = 0 if m == 0 else m / n
#     annotations.insert(0, f".")    
#     #annotations.append(f".")
#     annotations.insert(0, f"BL:{100 * percentage}%")
#     annotations = annotations[:4]
#     annotation_text = "\n".join(annotations)
#     plt.annotate(
#         annotation_text,
#         (input_size, pivot.loc[input_size, 1]),
#         textcoords="offset points",
#         xytext=(5,5),
#         ha='left',
#         fontsize=8,
#         bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.3)
#     )

# plt.title('Top 8 Configurations Throughput vs Input Size (RTX 4070)', fontsize=20)
# plt.xlabel('Input Size', fontsize=20)
# plt.ylabel('Throughput', fontsize=20)
# plt.legend()
# plt.grid(True)
# plt.tight_layout()

########################
# heatmap

# Simulate example structure of `df` for demonstration (replace with real df in actual use)
np.random.seed(0)
# df = pd.DataFrame({
#     'input_size': np.random.choice([2**i for i in range(14, 30)], size=1000),
#     'w': np.random.choice([32, 64, 128, 256, 512], size=1000),
#     't': np.random.choice([32, 64, 128, 256], size=1000),
#     'p': np.random.choice([0, 1], size=1000),
#     'b': np.random.choice(['a', 'c'], size=1000),
#     'a': 2,
#     's': np.random.choice([1, 8, 16], size=1000),
#     'm': np.random.choice([2, 4], size=1000),
#     'throughput': np.random.uniform(20, 40, size=1000)
# })

# 1. Define fixed parameter columns (w and t are variable)
# fixed_cols = ['p', 'b', 'a', 's', 'm']

# # 2. Define input size exponent ranges
# ranges = {
#     '2^14–2^18': list(range(14, 19)),
#     '2^19–2^22': list(range(19, 23)),
#     '2^23–2^29': list(range(23, 30)),
# }

# # 3. Find best-performing fixed params for each range
# best_configs = {}
# for name, exps in ranges.items():
#     inputs = [2**e for e in exps]
    
#     subset = df[df['input_size'].isin(inputs)]
#     print(df['input_size'])

#     if subset.empty:
#         print(f"Warning: No data found for input size range {name}")
#         continue

#     avg_by_fixed = (
#         subset.groupby(fixed_cols)['throughput']
#         .mean()
#         .reset_index()
#         .sort_values('throughput', ascending=False)
#     )

#     if avg_by_fixed.empty:
#         print(f"Warning: No fixed param groupings found for range {name}")
#         continue

#     best_config = avg_by_fixed.iloc[0][fixed_cols]
#     best_configs[name] = best_config

# # 4. Build 3x3 heatmap matrix: tuned_for x running_on
# heatmap_data = []
# for tuned_name, tuned_config in best_configs.items():
#     row = []
#     for running_name, exps in ranges.items():
#         inputs = [2**e for e in exps]
#         running_subset = df[df['input_size'].isin(inputs)]

#         # Mask for the tuned config
#         mask = pd.Series(True, index=running_subset.index)
#         for col in fixed_cols:
#             mask &= running_subset[col] == tuned_config[col]

#         avg_perf_tuned_on_running = running_subset[mask]['throughput'].mean()

#         # Best possible performance for this running range
#         best_perf = (
#             running_subset.groupby(fixed_cols)['throughput']
#             .mean()
#             .max()
#         )

#         # Compute ratio
#         ratio = avg_perf_tuned_on_running / best_perf if best_perf > 0 else np.nan
#         row.append(ratio)

#     heatmap_data.append(row)

# # 5. Create heatmap DataFrame
# heatmap_df = pd.DataFrame(
#     heatmap_data,
#     index=ranges.keys(),
#     columns=ranges.keys()
# )

# # 6. Plot the heatmap with reversed Y-axis and green-to-red color map
# plt.figure(figsize=(8, 6))
# sns.heatmap(
#     heatmap_df, annot=True, fmt=".2f",
#     cmap="RdYlGn",  # Green = good, Red = bad
#     xticklabels=True, yticklabels=True
# )
# plt.title("Throughput Ratio: Tuned For vs Running On")
# plt.xlabel("Running On")
# plt.ylabel("Tuned For")
# plt.gca().invert_yaxis()  # So 2^14–2^18 is at the bottom
# plt.tight_layout()
# plt.show()

#minimize distance between points.
#####################

# Ensure input data is already in `df`
# Columns required: input_size, throughput, w, t, a, s, b, p, m

# # Step 1: Compute the ideal line
# ideal_df = df.loc[df.groupby("input_size")["throughput"].idxmax()].sort_values("input_size")
# input_sizes = ideal_df["input_size"].tolist()
# ideal_throughputs = ideal_df["throughput"].tolist()
# print(input_sizes)
# print(ideal_throughputs)

# # Helper function to generate fixed-key
# def fixed_key(row):
#     return (row["a"], row["s"], row["b"], row["p"], row["m"])

# # Step 2: Group by fixed [a, s, b, p, m]
# grouped = df.groupby([df["a"], df["s"], df["b"], df["p"], df["m"]])

# candidates = []

# for fixed_params, group in grouped:
#     # Filter only rows with all required input_sizes
#     group_input_sizes = set(group["input_size"])
#     if not set(input_sizes).issubset(group_input_sizes):
#         continue

#     line_rows = []
#     for input_size in input_sizes:
#         subset = group[group["input_size"] == input_size]
#         best_row = subset.loc[subset["throughput"].idxmax()]
#         line_rows.append(best_row)

#     candidate_df = pd.DataFrame(line_rows)
#     candidate_throughputs = candidate_df["throughput"].tolist()

#     squared_diff = sum((np.array(candidate_throughputs) - np.array(ideal_throughputs))**2)
#     abs_diff = sum(abs(np.array(candidate_throughputs) - np.array(ideal_throughputs)))

#     candidates.append({
#         "fixed": fixed_params,
#         "df": candidate_df,
#         "squared_diff": squared_diff,
#         "abs_diff": abs_diff
#     })

# # Step 3: Select best candidates
# best_squared = min(candidates, key=lambda x: x["squared_diff"])
# best_abs = min(candidates, key=lambda x: x["abs_diff"])

# # Step 4: Plotting
# plt.figure(figsize=(14, 7))

# # Plot ideal line
# plt.plot(input_sizes, ideal_throughputs, '--', color='black', label='Ideal (Max Throughput)')

# # Plot best squared diff candidate
# squared_df = best_squared["df"]
# plt.plot(input_sizes, squared_df["throughput"], '-o', label='Best (Squared Diff)', color='blue')
# for i, row in squared_df.iterrows():
#     pass
#     #label = f"w={row.w} t={row.t} a={row.a} s={row.s} b={row.b} p={row.p} m={row.m} thr={row.throughput:.2f}"
#     #plt.annotate(label, (row.input_size, row.throughput), fontsize=8, color='blue', xytext=(0,5), textcoords='offset points')

# # Plot best absolute diff candidate
# abs_df = best_abs["df"]
# plt.plot(input_sizes, abs_df["throughput"], '-o', label='Best (Abs Diff)', color='red')
# for i, row in abs_df.iterrows():
#     pass
#     #label = f"w={row.w} t={row.t} a={row.a} s={row.s} b={row.b} p={row.p} m={row.m} thr={row.throughput:.2f}"
#     #plt.annotate(label, (row.input_size, row.throughput), fontsize=8, color='red', xytext=(0,-10), textcoords='offset points')

# # Labels and legend
# plt.xlabel("Input Size")
# plt.ylabel("Throughput")
# plt.title("Throughput vs Input Size\nIdeal Line vs Best Fixed Configurations")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()

# minimize area between curves
############
# --- Step 1: Create Ideal Line ---
ideal_line = df.groupby("input_size")["throughput"].max()
input_sizes = sorted(ideal_line.index.tolist())

# --- Step 2: Build Best Line per [a, s, b, p, m] ---
grouped_lines = {}
for group_key, group_df in df.groupby(["a", "s", "b", "p", "m"]):
    best_line = group_df.groupby("input_size")["throughput"].max()
    
    # Error if any input size is missing
    if set(input_sizes) != set(best_line.index):
        raise ValueError(f"Missing input sizes for group {group_key}")
    
    grouped_lines[group_key] = best_line.sort_index()

# --- Step 3: Compute Comparison Metrics ---
def compute_metrics(group_line):
    abs_diff = np.abs(group_line.values - ideal_line.values).mean()
    sqr_diff = ((group_line.values - ideal_line.values) ** 2).mean()
    area = np.trapz(np.abs(group_line.values - ideal_line.values), x=input_sizes)
    total_throughput = group_line.values.sum()
    return abs_diff, sqr_diff, area, total_throughput

scores = []
for key, line in grouped_lines.items():
    abs_diff, sqr_diff, area, total_tp = compute_metrics(line)
    scores.append({
        "key": key,
        "line": line,
        "abs": abs_diff,
        "sqr": sqr_diff,
        "area": area,
        "sum_throughput": total_tp

    })

scores_df = pd.DataFrame(scores)

# --- Step 4: Get Best Lines by Metric ---
best_abs = scores_df.loc[scores_df["abs"].idxmin()]
best_sqr = scores_df.loc[scores_df["sqr"].idxmin()]
best_area = scores_df.loc[scores_df["area"].idxmin()]
best_sum = scores_df.loc[scores_df["sum_throughput"].idxmax()]

# --- Step 5: Plot ---
plt.figure(figsize=(10, 6))
plt.axhline(y=800, color='r', linestyle='--', linewidth=2, label='Theoretical Bandwidth = 800 GB/s')
plt.plot(input_sizes, ideal_line.values, label="Ideal Line", color="black", linewidth=2)

plt.plot(input_sizes, best_abs["line"].values, label=f"Best Abs", linestyle='--')
plt.plot(input_sizes, best_sqr["line"].values, label=f"Best Sqr", linestyle='-.')
plt.plot(input_sizes, best_sum["line"].values, label=f"Best Sum", linestyle=':')

#plt.plot(input_sizes, best_area["line"].values, label=f"Best Area", linestyle=':')

plt.xlabel("Input Size", fontsize=16)
plt.ylabel("Throughput", fontsize=16)
plt.title("Best Group Lines vs Ideal Line", fontsize=16)
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig("stink local_scan.png")


