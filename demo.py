import numpy as np
import LinearRegression as lr
import LogisticRegression as lo
from sklearn import datasets

# Linear Regression

X,y = datasets.make_regression(n_features=1,n_informative=1, noise=20, random_state=1)
table=np.column_stack((X,y))

p = lr.LinearRegression(table,reg=True,lamda=10)
p.gradient_descent(1000,0.01)
print p.accuracy()
p.plot_fit(1)

# Logistic Regression Classification

x,y = datasets.make_classification(n_features=2, n_redundant=0, n_informative=2,n_clusters_per_class=1)
table = np.genfromtxt("datasets/toy.csv",delimiter=",")

table=np.column_stack((x,y))

p = lo.LogisticRegression(table,reg=True,lamda=10)
print p.gradient_descent(10000,0.03)
print p.accuracy()
p.plot_fit()


