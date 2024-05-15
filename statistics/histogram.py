import matplotlib.pyplot as plt
import numpy as np

# Read the data from the text file
with open('/content/question1_Normal_ChickenRabbit.txt', 'r') as file:
    lines = file.readlines()

# Parse the data into lists of x-values and y-values
seed = []
iterations = []
for line in lines:
    parts = line.strip().split(', ')  # Assuming comma-separated values, change ',' to your delimiter
    seed.append(float(parts[0]))
    iterations.append(float(parts[1]))

# Plot histogram
plt.hist(iterations, bins=10, color='blue', edgecolor='black')
plt.xlabel('iteration')
plt.ylabel('number of occurence')
plt.grid(True)
plt.savefig('1-1_Normal_GCD.png')

# Calculate mean
mean = np.mean(iterations)
print("Mean:", mean)

# Calculate variance
variance = np.var(iterations)
print("Variance:", variance)

# Print the count of iterations
count = len(iterations)
print("Total test count:", count)