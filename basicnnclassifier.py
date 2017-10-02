import numpy as np

fp=open("kd.txt")
input=[]
output=[]
for i in fp:
    i.rstrip()
    j=i.split()
    for k in j:
        k=float(k)
        print (type(k))

    output.append(j[-1])
    j.pop()
    input.append(j)
    print (j)

print (input)
print (output)

# sigmoid function
def nonlin(x,deriv=False):
    if(deriv==True):
        return x*(1-x)
    return 1/(1+np.exp(-x))

# input dataset
X = np.array(input,dtype=float)

# output dataset
y = np.array([output],dtype=float).T

# seed random numbers to make calculation
# deterministic (just a good practice)
np.random.seed(1)

# initialize weights randomly with mean 0
syn0 = 2*np.random.random((10,1)) - 1

for iter in range(10000):

    # forward propagation
    l0 = X
    l1 = nonlin(np.dot(l0,syn0))

    # how much did we miss?
    l1_error = y - l1

    # multiply how much we missed by the
    # slope of the sigmoid at the values in l1
    l1_delta = l1_error * nonlin(l1,True)
    print (l0.shape)
    print (l1_delta.shape)
    # update weights
    syn0 += np.dot(l0.T,l1_delta)
print (l1)