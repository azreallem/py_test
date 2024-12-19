import numpy as np
import pylab
import scipy.stats as stats
import pandas as pd
import matplotlib.pyplot as plot

# https://numpy.org/doc/stable/user/absolute_beginners.html
# https://docs.scipy.org/doc/scipy/tutorial/index.html#user-guide

def mytest01():
    data = np.mat([[1, 200, 105, 3, False],
                   [2, 165,  80, 2, False],
                   [3, 184.5, 120, 2, False],
                   [4, 116, 70.8, 1, False],
                   [5, 270, 150, 4, False]])
    row = 0
    for line in data:
        row += 1

    print(row)        # 5
    print(data.size)  # 25
    print(data[0, 3]) # 3.0
    print(data[0, 4]) # 0.0
    print(data)

def mytest02():
    data = np.mat([[1, 200, 105, 3, False],
                   [2, 165,  80, 2, False],
                   [3, 184.5, 120, 2, False],
                   [4, 116, 70.8, 1, False],
                   [5, 270, 150, 4, False]])
    mydata = []
    for row in data:
        mydata.append(row[0,1])

    print(mydata)
    print(np.sum(mydata))
    print(np.mean(mydata))
    print(np.std(mydata))
    print(np.var(mydata))

def mytest03():
    data = np.mat([[1, 200, 105, 3, False],
                   [2, 165,  80, 2, False],
                   [3, 184.5, 120, 2, False],
                   [4, 116, 70.8, 1, False],
                   [5, 270, 150, 4, False]])
    mydata = []
    for row in data:
        mydata.append(row[0,1])

    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.probplot.html#scipy.stats.probplot
    # Calculate quantiles for a probability plot, and optionally show the plot.
    stats.probplot(mydata, plot=pylab, rvalue=True)
    pylab.show()


def mytest04():
    rocksVMines = pd.DataFrame([[1, 200, 105, 3, False],
                         [2, 165,  80, 2, False],
                         [3, 184.5, 120, 2, False],
                         [4, 116, 70.8, 1, False],
                         [5, 270, 150, 4, False]])
    dataRow1 = rocksVMines.iloc[1, 0:3]
    dataRow2 = rocksVMines.iloc[2, 0:3]
    dataRow3 = rocksVMines.iloc[3, 0:3]

    plot.scatter(dataRow1, dataRow2)
    plot.xlabel("Attribute1")
    plot.ylabel("Attribute2")
    plot.show()

    plot.scatter(dataRow2, dataRow3)
    plot.xlabel("Attribute1")
    plot.ylabel("Attribute2")
    plot.show()




#----------------------------------------main----------------------------------------#
def main():
    mytest04()

if __name__ == '__main__':
    main()
