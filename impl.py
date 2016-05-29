import numpy as np
import linear_regression as lr
import logistic_regression as lo
from sklearn import datasets

"""
x,y = datasets.make_regression(n_features=1,n_informative=1, noise=20, random_state=0)
table=np.column_stack((x,y))

p = lr.linear_regression(table)
p.gradient_descent(100000,0.001)
p.plot_fit()
"""
#Logistic Regression

x,y = datasets.make_classification(n_features=2, n_redundant=0, n_informative=2,n_clusters_per_class=1)
table=np.column_stack((x,y))

p = lo.logistic_regression(table)
print p.gradient_descent(100,0.01)
p.plot_data()
p.plot_fit()
if p.predict(np.array([0,0]))>0.5:
	print 1
else:
	print 0
