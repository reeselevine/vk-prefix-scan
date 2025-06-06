import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches



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
print(df)


# # Group by workgroup size
# grouped = df.groupby("w")



# # Mean throughput per workgroup size
# print(grouped["throughput"].mean())



# grouped["throughput"].mean().plot(kind='bar')
# plt.title("Mean Throughput by Workgroup Size")
# plt.xlabel("Workgroup Size (-w)")
# plt.ylabel("Throughput")
# plt.savefig(output_file + ".png")


# Top 8 fixed configs
################################

# # fixed params columns
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
# n = 8
# top_fixed_params = avg_throughput_by_fixed.iloc[:n]

# # Storage for both sets
# lines = []
# i = 0
# for rank, fixed_row in top_fixed_params.iterrows():
#     # Get rows in df with matching fixed params
#     mask = pd.Series(True, index=df.index)
#     for col in fixed_cols:
#         mask &= df[col] == fixed_row[col]

#     filtered_df = df[mask]

#     # Find max throughput per input size (best w,t for each input size)
#     best_per_input = filtered_df.loc[
#         filtered_df.groupby('input_size')['throughput'].idxmax()
#     ][['input_size', 'throughput', 'w', 't']].sort_values('input_size')

#     label = f"Rank {i + 1}: " + ", ".join(f"{k}={fixed_row[k]}" for k in fixed_cols)
#     lines.append((best_per_input, label))
#     i += 1

# # 4. Plot both
# plt.figure(figsize=(10,6))

# for line_data, label in lines:
#     label = label.replace("a=2, ", "")
#     plt.plot(line_data['input_size'], line_data['throughput'], marker='o', label=label)
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
# plt.title("Top 8 Fixed Param Sets: Throughput vs Input Size")
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


# Best avg reduction per workgroup size
################################

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

#################################3

# For each input_size, get top 3 configs by throughput
top3_per_input = (
    df
    .sort_values(['input_size', 'throughput'], ascending=[True, False])
    .groupby('input_size')
    .head(3)
    .copy()
)

# Assign rank per input size (1=best)
top3_per_input['rank'] = top3_per_input.groupby('input_size')['throughput'].rank(ascending=False, method='first')

# Pivot so each rank is a separate line, index=input_size, columns=rank, values=throughput
pivot = top3_per_input.pivot(index='input_size', columns='rank', values='throughput')

plt.figure(figsize=(12, 7))

# Plot lines for rank 1, 2, 3
for rank in [1, 2, 3]:
    if rank in pivot.columns:
        plt.plot(pivot.index, pivot[rank], marker='o', label=f'Top {int(rank)} config')

# Annotate each input size with all three configs' summaries
for input_size in pivot.index:
    configs = top3_per_input[top3_per_input['input_size'] == input_size]
    # Build annotation string for all three configs at this input size
    annotations = []
    for _, row in configs.iterrows():
        # Customize this summary as you like, here showing w, t, b
        annotations.append(f"(w={row['w']}, t={row['t']}, b={row['b']})")
    annotation_text = "\n".join(annotations)
    plt.annotate(
        annotation_text,
        (input_size, pivot.loc[input_size, 1]),
        textcoords="offset points",
        xytext=(5,5),
        ha='left',
        fontsize=8,
        bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.3)
    )

plt.title('Top 3 Configurations Throughput vs Input Size')
plt.xlabel('Input Size')
plt.ylabel('Throughput')
plt.legend()
plt.grid(True)
plt.tight_layout()


plt.savefig("stink local_scan.png")


