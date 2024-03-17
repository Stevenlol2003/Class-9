import csv
import matplotlib.pyplot as plt
import sys
import numpy as np

def main():
    file = sys.argv[1] # the first argument as string
    # file = 'hw5.csv' # use for VS code only, switch to line above when submitting
    # file = 'toy.csv' # use for VS code only, switch to line above when submitting

    years = []
    days = []

    with open(file) as f:
        data = csv.DictReader(f)
        for row in data:
            years.append(int(row['year']))
            days.append(int(row['days']))

    plt.figure()
    plt.plot(years, days)
    plt.xlabel("Year")
    plt.ylabel("Number of Frozen Days")
    plt.savefig("plot.jpg") # must be plot.jpg

    # Q3a
    onesArray = np.ones(len(years), dtype=np.int64)
    # print(onesArray)
    yearsArray = np.array(years, dtype=np.int64)
    # print(yearsArray)
    X = np.array([onesArray, yearsArray], dtype=np.int64).T
    print("Q3a:")
    print(X) # Your X.dtype needs to be int64

    # Q3b
    Y = np.array(days, dtype=np.int64)
    print("Q3b:")
    print(Y ) # Your Y.dtype needs to be int64

    # Q3c
    Z = (X.T @ X).astype(np.int64)
    print("Q3c:")
    print(Z) # Your Z.dtype needs to be int64

    # Q3d
    # Source of linalg.inv, https://numpy.org/doc/stable/reference/generated/numpy.linalg.inv.html
    I = np.linalg.inv(Z)
    print("Q3d:")
    print(I)

    # Q3e
    PI = I @ (X.T)
    print("Q3e:")
    print(PI)

    # Q3f
    β = PI @ Y
    print("Q3f:")
    print(β)

    # Q4
    xtest = 2022
    yhattest = β[0] + β[1] * xtest
    print("Q4: " + str(yhattest))

    # Q5a
    sign = "=" # default of =, where value is 0
    if β[1] > 0:
        sign = ">" # if value is postive, set sign to >
    elif β[1] < 0:
        sign = "<" # if value is negative, set sign to <
    print("Q5a: " + sign)

    # Q5b
    explanation5b = "If the symbol is >, it means that the slope of the regression is positive;\nif the symbol is <, it means that the slope of the regression is negative;\nif the symbol is 0, it means that the slope of the regression is 0 (a horizontal line)"
    print("Q5b: " + explanation5b)

    # Q6a
    # formula is 0 = β[0] + β[1] * x*
    # x* = (0 - β[0]) / β[1]
    estimated_year = (0 - β[0]) / β[1]
    estimated_year_string = estimated_year.astype(str)
    print("Q6a: " + estimated_year_string)

    # Q6b
    explanation6b = "The x* value makes sense, because β[1] is negative,\nmeaning that the slop of the regression is negative, \neventually the x value (years) would reach 0.\nBased off the in plot.jpg,\nwe can see that overtime number of frozen days is decreasing,\nthe year 2455 is a reasonable prediction to which Lake Mendota will no longer freeze."
    print("Q6b: " + explanation6b)

if __name__ == "__main__":
    main()