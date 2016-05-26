import numpy as np
import linear_regression as lr

x = np.loadtxt("datasets/ex2x.dat")
y= np.loadtxt("datasets/ex2y.dat")
table=np.column_stack((x,y))
p = lr.linear_regression(table)
p.plot_data()
p.gradient_descent(500,0.01)
p.plot_fit()
