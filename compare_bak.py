__author__ = 'alex'

from PyAI import *
from pandas import HDFStore
import h5py
import dask.array as da
from utils import dask_cov

# Gather data from databases

with h5py.File('close.hdf5') as f:
    close_d = f['/quotes/close']
    open_d = f['/quotes/open']

    close = da.from_array(close_d, chunks=(100, 100))
    open = da.from_array(open_d, chunks=(100, 100))

    var = close - open

    # Standardize data
    var -= var.mean(axis=0)
    var /= var.std(axis=0)
    var = var.T

    # Create unsupervised labels
    labels = cluster.AffinityPropagation().fit_predict(dask_cov(var).compute())
    store = HDFStore('quotes.hdf5')
    names = store.quotes.minor_axis

    brain = Brain(var.compute(), labels)
    data = var.compute()

brain.init_data_transformation(TRANSFORMATION.Standardize(), TRANSFORMATION.PCA.RandomizedPCA())
brain.init_naive_bayes(NAIVE_BAYES.REGULAR)


