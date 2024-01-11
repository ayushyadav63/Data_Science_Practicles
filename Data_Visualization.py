x=np.array([[1,2,],[5,10,]])
print(x)
y=np.array([[3,4],[6,9]])
print(y)
#[[ 1  2]
 [ 5 10]]
[[3 4]
 [6 9]]


#adding the two matrixes
print('adding the following two matrixes:')
a=np.add(x,y)
print(a)
#adding the following two matrixes:
[[ 4  6]
 [11 19]]

#subtracting the two matrixes
print('subtracting the following two matrixes:')
b=np.subtract(x,y)
print(b)
#subtracting the following two matrixes:
[[-2 -2]
 [-1  1]]

#multiplying the two matrixes
print('multiply the following two matrixes:')
c=np.multiply(x,y)
print(c)
#multiply the following two matrixes:
[[ 3  8]
 [30 90]]


#dividing the two matrixes
print('dividing the following two matrixes:')
d=np.divide(x,y)
print(d)
#dividing the following two matrixes:
[[0.33333333 0.5       ]
 [0.83333333 1.11111111]]

#dot the two matrixes
print('dot the two matrixes')
e=np.dot(x,y)
print(e)
#dot the two matrixes
[[ 15  22]
 [ 75 110]]

#squareroot of the two matrixes
print('the square root of two matrixes are')
f=np.sqrt(y)
print(f)
#the square root of two matrixes are
[[1.73205081 2.        ]
 [2.44948974 3.        ]]

#sum axis-wise of the two matrixes
print('sum of matrix axis-wise')
g=np.sum(x,axis=0)
print(g)
#sum of matrix axis-wise
[ 6 12]

#transpose of matrix
print('transpose of matrix')
print(x.T)
#transpose of matrix
[[ 1  5]
 [ 2 10]]


                     #practicle no -2

#first import thses libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

x=[1,2,3,4,5,6,7]
y=[2,3,6,7,8,9,6]
plt.plot(x, y)
plt.show()

x=np.array([3,8,1,10])
plt.plot(x, marker='o')
plt.show()

df=pd.read_csv('https://raw.githubusercontent.com/codeforamerica/ohana-api/master/data/sample-csv/addresses.csv')
print(df)
plt.plot(df['city'],df['postal_code'])
plt.show()

plt.scatter(df['city'],df['postal_code'])
plt.xlabel=('x-axis')
plt.ylabel=('y-axis')
plt.show()

#pie plotting
y=[8,3,1,10]
mylabels=['apples','bananas','cheries','dates']
myexplode=[0.2,0.1,0,0]
plt.pie(y,labels=mylabels,explode=myexplode)
plt.show()


df=pd.read_csv('https://raw.githubusercontent.com/codeforamerica/ohana-api/master/data/sample-csv/addresses.csv')
print(df)
df1=df.plot.box()
plt.boxplot(df1['postal_code'])
plt.show()



