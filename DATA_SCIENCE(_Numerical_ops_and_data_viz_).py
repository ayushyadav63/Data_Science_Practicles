#Practicle n0=1

# Matrix Operations
import numpy as np

# Matrix Definitions
x = np.array([[1, 2], [5, 10]])
y = np.array([[3, 4], [6, 9]])

# Display matrices
print("Matrix X:")
print(x)
print("Matrix Y:")
print(y)

# Matrix Addition
print('\nAdding the following two matrices:')
a = np.add(x, y)
print(a)

# Matrix Subtraction
print('\nSubtracting the following two matrices:')
b = np.subtract(x, y)
print(b)

# Matrix Multiplication
print('\nMultiplying the following two matrices:')
c = np.multiply(x, y)
print(c)

# Matrix Division
print('\nDividing the following two matrices:')
d = np.divide(x, y)
print(d)

# Dot Product of Matrices
print('\nDot product of the following two matrices:')
e = np.dot(x, y)
print(e)

# Square Root of Matrix Y
print('\nThe square root of matrix Y:')
f = np.sqrt(y)
print(f)

# Sum Axis-wise of Matrix X
print('\nSum of matrix X axis-wise:')
g = np.sum(x, axis=0)
print(g)

# Transpose of Matrix X
print('\nTranspose of matrix X:')
print(x.T)


# Practical No - 2: Data Visualization

# Importing libraries
import matplotlib.pyplot as plt
import pandas as pd

# Line Plot
x = [1, 2, 3, 4, 5, 6, 7]
y = [2, 3, 6, 7, 8, 9, 6]
plt.plot(x, y)
plt.show()

# Line Plot with Marker
x = np.array([3, 8, 1, 10])
plt.plot(x, marker='o')
plt.show()

# Reading CSV and Scatter Plot
df = pd.read_csv('https://raw.githubusercontent.com/codeforamerica/ohana-api/master/data/sample-csv/addresses.csv')
print(df)
plt.scatter(df['city'], df['postal_code'])
plt.xlabel('x-axis')
plt.ylabel('y-axis')
plt.show()

# Pie Plotting
y = [8, 3, 1, 10]
mylabels = ['apples', 'bananas', 'cherries', 'dates']
myexplode = [0.2, 0.1, 0, 0]
plt.pie(y, labels=mylabels, explode=myexplode)
plt.show()

# Box Plot
df = pd.read_csv('https://raw.githubusercontent.com/codeforamerica/ohana-api/master/data/sample-csv/addresses.csv')
print(df)
df.plot.box()
plt.boxplot(df['postal_code'])
plt.show()
