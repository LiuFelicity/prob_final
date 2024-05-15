import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import math

# Read the data from the text file
with open('/content/question1_Normal_ChickenRabbit.txt', 'r') as file:
    lines = file.readlines()

iterations_Normal = []
for line in lines:
    parts = line.strip().split(', ')
    iterations_Normal.append(float(parts[1]))

with open('/content/question1_Xavier_ChickenRabbit.txt', 'r') as file:
    lines = file.readlines()

iterations_Xavier = []
for line in lines:
    parts = line.strip().split(', ')
    iterations_Xavier.append(float(parts[1]))

# Plot histogram
plt.hist(iterations, bins=10, color='blue', edgecolor='black')
plt.xlabel('iteration')
plt.ylabel('number of occurence')
plt.grid(True)
plt.savefig('1-1_Normal_GCD.png')

# Calculate mean
Normal_u = np.mean(iterations_Normal)
Xavier_u = np.mean(iterations_Xavier)

# Calculate variance
Normal_var = np.var(iterations_Normal)
Xavier_var = np.var(iterations_Xavier)


# Print the count of iterations
n = len(iterations_Normal)
m = len(iterations_Xavier)


t_score = (Normal_u - Xavier_u)/(pow((Normal_var/(n-1))+(Xavier_var/(m-1)), 0.5)) # 究竟是n-1還是n
df = math.floor(((Normal_var/n + Xavier_var/m)**2) / (((Normal_var/n)**2)/(n-1)+((Xavier_var/m)**2)/(m-1)))
print("t score: ", t_score)
print("fd: ", df)
# 計算對應的 p-value
p_value = stats.t.sf(t_score, df)

print("t score 對應的 p-value:", p_value)