import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt

# Store the voltage and current data as column vectors.
I = np.mat([0.2, 0.3, 0.4, 0.5, 0.6]).T
V = np.mat([1.23, 1.38, 2.06, 2.47, 3.17]).T


plt.scatter(np.asarray(I), np.asarray(V))

plt.xlabel('Current (A)')
plt.ylabel('Voltage (V)')
plt.grid(True)
plt.show()



# Define the H matrix, what does it contain?
H = I 		# V = I*R ==> Y = V, R = X, I = H

# Now estimate the resistance parameter.
R = np.linalg.inv(H.transpose()*H)*(H.transpose())*(V)

print('The slope parameter (i.e., resistance) for the best-fit line is:')
print(R)



# Define the H matrix, what does it contain?
H = I

# Now estimate the resistance parameter.
R = np.linalg.inv(H.transpose()*H)*(H.transpose())*(V)

print('The slope parameter (i.e., resistance) for the best-fit line is:')
print(R)

