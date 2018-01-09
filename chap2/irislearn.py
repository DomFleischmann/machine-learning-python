import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np 
from perceptron import Perceptron
from matplotlib.colors import ListedColormap

def plot_irisdata(X, y):
    plt.scatter(X[:50, 0], X[:50, 1],
        color='red', marker='o', label='setosa')
    plt.scatter(X[50:100, 0], X[50:100, 1],
        color='blue', marker='x', label='versicolor')
    plt.xlabel('sepal length')
    plt.ylabel('petal length')
    plt.legend(loc='upper left')
    plt.show()

def plot_misclassifications(ppn):
    plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Number of misclassifications')
    plt.show()

def plot_decision_regions(X, y, classifier, resolution=0.02):
    # setup marker generator and color map
    markers = ('s','x','o','^','v')
    colors = ('red','blue','lightgreen','gray','cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    #plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arrange(x1_min, x1_max, resolution),
                           np.arrange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    #plot class samples
    #TODO

df = pd.read_csv('iris.data', header=None)
print(df.tail())

y = df.iloc[0:100, 4].values

y = np.where(y == 'Iris-setosa', -1, 1)
X = df.iloc[0:100, [0, 2]].values
plot_irisdata(X, y)

ppn =Perceptron(eta=0.1, n_iter=10)
ppn.fit(X,y)
plot_misclassifications(ppn)


