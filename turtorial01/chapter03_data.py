import matplotlib.cm
import numpy as np
import pylab
from pylab import *
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

def mytest05():
    filePath = ("dataTest.csv")
    dataFile = pd.read_csv(filePath, header=None)

    dataRow1 = dataFile.iloc[100, 1:300]    # [row, col]
    dataRow2 = dataFile.iloc[101, 1:300]
    plot.scatter(dataRow1, dataRow2)
    plot.xlabel("Attribute1")
    plot.ylabel("Attribute2")
    plot.show()

def mytest06():
    filePath = ("dataTest.csv")
    dataFile = pd.read_csv(filePath, header=None)

    target = []
    for i in range(200):
        if dataFile.iloc[i, 10] >= 7:
            target.append(1.0)
        else:
            target.append(0.0)
    dataRow = dataFile.iloc[0:200, 10]
    plot.scatter(dataRow, target)
    plot.xlabel("Attribute")
    plot.ylabel("Target")
    plot.show()

def mytest07():
    filePath = ("dataTest.csv")
    dataFile = pd.read_csv(filePath, header=None)

    target = []
    for i in range(200):
        if dataFile.iloc[i, 10] >= 7:
            target.append(1.0 + np.random.uniform(-0.3, 0.3))
        else:
            target.append(0.0 + np.random.uniform(-0.3, 0.3))
    dataRow = dataFile.iloc[0:200, 10]
    plot.scatter(dataRow, target, alpha=0.5, s=100)
    plot.xlabel("Attribute")
    plot.ylabel("Target")
    plot.show()

def mytest08():
    filePath = ("dataTest.csv")
    dataFile = pd.read_csv(filePath, header=None)

    print(dataFile.head())
    print(dataFile.tail())

    summary = dataFile.describe()
    print(summary)

    array = dataFile.iloc[:,10:16].values

    boxplot(array)
    xlabel("Attribute")
    ylabel("Score")
    show()

def mytest09():
    filePath = ("dataTest.csv")
    dataFile = pd.read_csv(filePath, header=None)

    summary = dataFile.describe()
    dataFileNormalized = dataFile.iloc[:,1:6]
    for i in range(5):
        mean = summary.iloc[1, i]
        sd = summary.iloc[2, i]

    dataFileNormalized.iloc[:, i:(i+1)] = (dataFileNormalized.iloc[:, i:(i+1)] - mean) / sd
    array = dataFileNormalized.values

    boxplot(array)
    xlabel("Attribute")
    ylabel("Score")
    show()

def mytest10():
    filePath = ("dataTest.csv")
    dataFile = pd.read_csv(filePath, header=None)

    summary = dataFile.describe()
    minRings = -1
    maxRings = 99
    nrows = 10
    for i in range(nrows):
        dataRow = dataFile.iloc[i, 1:10]
        labelColor = (dataFile.iloc[i, 10] - minRings) / (maxRings - minRings)
        dataRow.plot(color=matplotlib.cm.RdYlBu(labelColor), alpha=0.5)

    xlabel("Attribute")
    ylabel("Score")
    show()


def mytest11():
    filePath = ("dataTest.csv")
    dataFile = pd.read_csv(filePath, header=None)

    summary = dataFile.describe()
    corMat = pd.DataFrame(dataFile.iloc[1:20, 1:20].corr())

    plot.pcolor(corMat)
    show()

def mytest12():
    filePath = ("rain.csv")
    dataFile = pd.read_csv(filePath, header=None)
    summary = dataFile.describe()
    print(summary)
    array = dataFile.iloc[:,1:13].values
    boxplot(array)
    plot.xlabel("month")
    plot.ylabel("rain")
    show()

def mytest13():
    filePath = ("rain.csv")
    dataFile = pd.read_csv(filePath, header=None)
    summary = dataFile.describe()
    minRings = -1
    maxRings = 99
    nrows = 11
    for i in range(nrows):
        dataRow = dataFile.iloc[i, 1:13]
        labelColor = (dataFile.iloc[i, 12] - minRings) / (maxRings - minRings)
        dataRow.plot(color=plot.cm.RdYlBu(labelColor), alpha=0.5)
    plot.xlabel("Attribute")
    plot.ylabel("Score")
    show()

def mytest14():
    filePath = ("rain.csv")
    dataFile = pd.read_csv(filePath)
    summary = dataFile.describe()
    corMat = pd.DataFrame(dataFile.iloc[1:20, 1:20].corr())
    plot.pcolor(corMat)
    plot.show()


#----------------------------------------main----------------------------------------#
def main():
    mytest14()

if __name__ == '__main__':
    main()
