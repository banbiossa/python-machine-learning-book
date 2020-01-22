import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.perceptron import Perceptron, AdalineGD

from matplotlib.colors import ListedColormap

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')


def plot_decision_regions(X, y, classifier, resolution=.02):
    # setup marker generator and color map
    markers = list('sxo^v')
    colors = 'red blue lightgreen gray cyan'.split()
    cmap = ListedColormap(colors[:len(np.unique(y))])
    
    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(
        np.arange(x1_min, x1_max, resolution),
        np.arange(x2_min, x2_max, resolution),
    )
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    
    # plot class samples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(
            x=X[y==cl, 0],
            y=X[y==cl, 1],
            alpha=.8, 
            c = colors[idx],
            marker = markers[idx],
            label = cl,
            edgecolor='black'
        )

def load_iris():
    df = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data", header=None)
    df.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'iris']

    
    