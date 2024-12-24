import numpy as np
import sys
from numpy import pi
from numpy import newaxis

#https://numpy.org/doc/2.1/user/quickstart.html

#ndarray

#An example
def mytest01():
    a = np.arange(15).reshape(3, 5)    # 2d array
    print(a, a.shape, a.ndim, a.dtype.name, a.itemsize, a.size, type(a))
    
    b = np.array([6, 7, 8])
    print(b, type(b))

# Array creation
def mytest02():
    a = np.array([3, 4, 5])
    a = np.array([1.2, 3.5, 5.1])
    a = np.array([1.2, 3.5, 5.1], dtype=int)
    a = np.array([[1.5,2,3], [4,5,6]])

    a = np.zeros((3, 4))
    a = np.ones((3, 4), dtype=np.int16)
 
    a = np.arange(10, 30, 5)    # [10,30) step: 5
    a = np.arange(0, 2, 0.3)    # [0, 2) step: 0.3

    a = np.linspace(0, 2, 9)    # [0,2] num: 9
    a = np.linspace(0, 2*pi, 100)
    a = np.sin(a)

    print(a, a.dtype)

# Printing arrays
def mytest03():
    a = np.arange(8)    # 1d array
    a = np.arange(12).reshape(4, 3)    # 2d array
    a = np.arange(24).reshape(2, 3, 4)    # 3d array
    #print(a, a.dtype)

    # print the entire array
    np.set_printoptions(threshold=sys.maxsize)
    print(np.arange(10000))

# Basic operations
def mytest04():
    a = np.array([20, 30, 40, 50])
    b = np.arange(4)
    c = a - b
    b = b**2
    a = 10 * np.sin(a)
    a = a < 35

    A = np.array([[1, 1],
                  [0, 1]])
    B = np.array([[2, 0],
                  [3, 4]])
    C = A * B
    D = A @ B    # anther func: A.dot(B)
    #print(C)
    #print(D)

    rg = np.random.default_rng(1) # create instance of default random number generator
    a = np.ones((2, 3), dtype=int)
    b = rg.random((2, 3))
    a *= 3
    b += a
    #print(b)
    #a += b    # Error: cannot cast from float to int

    a = rg.random((2, 3))
    b = a.sum()
    c = a.min()
    d = a.max()
    #print(a, b, c ,d)

    a = np.arange(12).reshape(3, 4)
    b = a.sum(axis=0)    # sum of each col
    c = a.min(axis=1)    # sum of each row
    d = a.cumsum(axis=1) # cumulative sum along each row
    print(a, b, c ,d)

# Universal functions
def mytest05():
    B = np.arange(3)
    B1 = np.exp(B)
    B2 = np.sqrt(B)
    C = np.array([2.,-1.,4.])
    C1 = np.add(B, C)
    print(B, B1, B2, C, C1)

# Indexing, slicing and iterating
def myfunc01(x, y):
    return 10 * x + y

def mytest06():
    a = np.arange(10)**3 
    #print(a, a[2], a[2:5])    # [2,5)
    a[:6:2] = 1000    # from start to 6, step 2
    #print(a)
    a = a[::-1]    # reversed a
    #print(a)

    b = np.fromfunction(myfunc01, (5, 4), dtype=int)
    c = b[2, 3]
    d = b[0:5, 1]
    e = b[:, 1]
    f = b[1:3, :]
    g = b[-1]
    #print(b, c, d, e, f, g)

    c = np.array([[[  0,  1,  2],  # a 3D array (two stacked 2D arrays)
                   [ 10, 12, 13]],
                  [[100, 101, 102],
                   [110, 112, 113]]])
    d = c[1, ...]
    e = c[..., 2]
    #print(c, c.shape, d, e)

    for row in b:
        print(row)
    for element in b.flat:
        print(element)

# Shape manipulation

# Chaning the shape of an array
def mytest07():
    rg = np.random.default_rng(1) # create instance of default random number generator
    a = np.floor(10 * rg.random((3, 4)))
    print(a, a.shape)
    b = a.ravel() # returns the array, flattened
    c = a.reshape(6, 2)
    d = a.T # returns the array, transposed
    #print(b, c, d)

    # The reshape function returns its argument with a modified shape,
    # whereas the ndarray.resize method modifies the array itself.
    a.resize((2, 6))
    print(a)

    # If a dimension is given as -1 in a reshaping operation,
    # the other dimensions are automatically calculated:
    a = a.reshape(3, -1)
    print(a)

# Stacking together different arrays
def mytest08():
    rg = np.random.default_rng(1)
    a = np.floor(10 * rg.random((2, 2)))
    b = np.floor(10 * rg.random((2, 2)))
    c = np.vstack((a, b))
    d = np.hstack((a, b))
    #print(a, b, c, d)

    c = np.column_stack((a, b))  # returns a 2D array
    #print(c)

    a = np.array([4., 2.])
    b = np.array([3., 8.])
    c = np.column_stack((a, b))  # returns a 2D array
    d = np.hstack((a, b))        # the result is different
    e = a[:, newaxis]            # view `a` as a 2D column vector
    f = np.column_stack((a[:, newaxis], b[:, newaxis])) # the result is the same `c`
    g = np.hstack((a[:, newaxis], b[:, newaxis]))  # the result is the same
    #print(a, b, c, d, e, f, g)

    a = np.r_[1:4, 0, 4]
    print(a)

# Splitting one array into serveral smaller ones
def mytest09():
    rg = np.random.default_rng(1)
    a = np.floor(10 * rg.random((2, 12)))
    b = np.hsplit(a, 3)
    c = np.hsplit(a, (3, 4))
    print(a, b, c)

 
# Copies and views
def myfunc02(x):
   print(id(x)) 

def mytest10():
    a = np.arange(12).reshape((3, 4))
    b = a
    if b is a:
        print(True)

    print(id(a))    # id is a unique identifier of an object 
    myfunc02(b)     # same

# View or shallow copy
def mytest11():
    a = np.arange(12).reshape((3, 4))
    c = a.view()
    print(True if c is a else False)
    print(True if c.base is a.base else False)
    print(c, c.base)
    print(True if c.flags.owndata else False)
    c = c.reshape((2, 6))    # a's shape doesn't change
    print(a.shape)
    c[0, 4] = 1234           # a's data changes
    print(a)

    s = a[:, 1:3]
    s[:] = 10
    print(a)                 # change

# Deep copy 
def mytest12():
    a = np.arange(12).reshape((3, 4))
    d = a.copy()
    print(True if d is a else False)
    print(True if d.base is a.base else False)
    print(a, a.base, d, d.base)
    d[0, 0] = 9999
    print(a)                # no change

    a = np.arange(int(1e8))
    print(a)                # no change
    b = a[:100].copy()
    del a  # the memory of ``a`` can be released.
    print(b)                # no change


#----------------------------------------main----------------------------------------#
def main():
    mytest12()

if __name__ == '__main__':
    main()
