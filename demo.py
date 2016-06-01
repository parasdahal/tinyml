import numpy as np
from mclearn import LinearRegression as lr
from mclearn import LogisticRegression as lo
from sklearn import datasets

# Linear Regression

X,y = datasets.make_regression(n_features=1,n_informative=1, noise=20, random_state=1)
table=np.column_stack((X,y))

p = lr.LinearRegression(table,reg=True,lamda=10)
#p.gradient_descent(1000,0.01)
#print p.accuracy()
#p.plot_fit()

# Logistic Regression Classification

x,y = datasets.make_classification(n_features=2, n_redundant=0, n_informative=2,n_clusters_per_class=1)
table = np.loadtxt("datasets/ex2data2.txt",delimiter=",")

#table=np.column_stack((x,y))

p = lo.LogisticRegression(table,reg=False,lamda=0,degree=3)
print p.gradient_descent(5000,0.01)
print p.accuracy()
p.plot_fit()
