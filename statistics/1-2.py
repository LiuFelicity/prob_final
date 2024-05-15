import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chi2

# Read the data from the text file
with open('/content/question2_GCD_seed1000.txt', 'r') as file:
    lines = file.readlines()

# Parse the data into lists of x-values and y-values
seed = []
iterations = []
for line in lines:
    parts = line.strip().split(', ')  # Assuming comma-separated values, change ',' to your delimiter
    iterations.append(float(parts[1]))

# Calculate mean
mean = np.mean(iterations)
print("Mean:", mean)

# Calculate variance
variance = np.var(iterations)
print("Variance:", variance)

# Print the count of iterations
n = len(iterations)

sigma = 150 #guess std

x = ((n-1)*variance)/((sigma)**2)
print("x: ",x)
p = chi2.cdf(x, n-1)
print("p-value:", 1-p)