from pandas import HDFStore
from utils import dask_cov
from PyAI import *


with HDFStore('test.hdf5') as f:
    Close = f.Close
    close = Close.as_matrix()
    open = f.Open.as_matrix()
    names = f.Close.columns.tolist()

    var = close - open
    var -= var.mean(axis=0)
    var /= var.std(axis=0)
    var = var.T

    # Create unsupervised labels
    labels = cluster.AffinityPropagation(0.7).fit_predict(dask_cov(var))
    brain = Brain(var, labels)

brain.init_data_transformation(TRANSFORMATION.Standardize(), TRANSFORMATION.PCA.RandomizedPCA())
brain.init_naive_bayes(NAIVE_BAYES.REGULAR)

