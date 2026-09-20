#!/usr/bin/env python3

import re
import matplotlib.pyplot as plt

x_vals = []
y_vals = []

with open('job.out', 'r') as file:
    for line in file:
        if 'E0=' in line:
            # Extract the first number in the line (before 'T=')
            match_x = re.match(r"\s*(\d+)", line)
            # Extract the number following 'E0='
            match_y = re.search(r"E0=\s*([\-\.\dE+]+)", line)

            if match_x and match_y:
                x = float(match_x.group(1))
                y = float(match_y.group(1))
                x_vals.append(x)
                y_vals.append(y)

# Use only the last n values
#x_vals = x_vals[-100:]
#y_vals = y_vals[-100:]


# Plotting
plt.figure(figsize=(10, 5))
plt.plot(x_vals, y_vals, marker='o', markersize=1, linestyle='-', color='blue')
plt.xlabel('Step Index')
plt.ylabel('E0 Energy Value')
plt.title('E0 vs Step Index')
plt.grid(True)
plt.tight_layout()
plt.savefig('e0_plot.svg')
