import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import sys
import path

# filename = "/home/nvidia/git/mt/out/identity-kernel-stream-1x1-25f-to-25000f-1c-grid-search-buffer-size-data.csv"
filename = path.out(sys.argv[1])

num_rows = 0
row_names = []
values = []
# Read the values from the file
with open(filename+".csv") as file:
    for line in file:
        num_rows += 1
        row_values = line.rstrip().split(",")
        row_names.append(row_values[0])
        values.extend(map(float, row_values[1:]))

# Determine the number of rows and columns
# Assuming you know the number of columns (e.g., 3 columns)
num_columns = len(values) // num_rows
print(f"num_values: {len(values)} num_columns: {num_columns}, num_rows: {num_rows}")
# Convert the list to a NumPy array and reshape it
array = np.array(values).reshape((num_rows, num_columns))

# Define the range of constants
# constants = np.linspace(np.min(array), np.max(array), 10000)
constants = np.arange(int(np.min(array)), int(np.max(array)), 1)

# Calculate the cumulative probability for each constant
cumulative_probabilities = np.zeros((num_rows, len(constants)))

for i, constant in enumerate(constants):
    cumulative_probabilities[:, i] = np.cumsum(array > constant, axis=1)[:, -1] / num_columns

# Define different marker styles for each row
markers = ['x', '1', '2', '3', '4']


# Plot the cumulative probability for each row as a scatter plot with dotted lines
for i in range(num_rows):
    # Filter out unchanged probabilities
    changes = np.diff(cumulative_probabilities[i], prepend=np.nan) != 0
    plt.scatter(constants[changes], cumulative_probabilities[i][changes], label=row_names[i], s=30, marker=markers[i % len(markers)], linewidth=1)  # s=1 sets the marker size
    plt.plot(constants[changes], cumulative_probabilities[i][changes], linestyle='--', linewidth=1)  # Add straight lines

plt.xlabel('Execution Time (us)')
plt.ylabel('Pr{T>=t}')
plt.ylim(1e-5, 1)  # Set y-axis range to [0, 1]
plt.xlim(10, 1e5)  # Set y-axis range to [0, 1]
# plt.xlim(0, 5000)  # Set y-axis range to [0, 1]
plt.xscale('log')  # Set y-axis to logarithmic scale
plt.yscale('log')  # Set y-axis to logarithmic scale
plt.grid(True, which="major", linestyle='-', linewidth=0.5)  # Add grid lines
plt.legend()
plt.savefig(filename + ".png", dpi=300)
