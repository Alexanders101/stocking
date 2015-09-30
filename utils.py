import googlefinance
from pandas.io.data import DataReader
import numpy as np
import datetime


def get_sp500(start_year=2015):
    from bs4 import BeautifulSoup
    import wikipedia
    import pandas
    import numpy as np

    wiki = wikipedia.page('List of S&P 500 companies')
    soup = BeautifulSoup(wiki.html())
    table = soup.find("table", {"class": "wikitable sortable"})
    df = pandas.read_html(str(table))
    df = df[0][[0, 1, 3, 6]][1:]

    # symbols, names, industry, date =

    df = np.array(
        [df[0].tolist(), df[1].tolist(), df[3].tolist(), df[6].tolist()])
    df[3][df[3] == 'nan'] = '1950-01-01'
    ind = [False if int(year[:4]) >= start_year else True for year in df[3]]
    df = df[:, np.where(ind)[0]]

    return (df)


def get_stock(symbol, start=None, stop=None, array=False):
    ts = DataReader(symbol, 'yahoo', start, stop)
    if array:
        return ts.as_matrix()
    return ts


def get_stock_live(symbol, just_price=False):
    rt = googlefinance.getQuotes(symbol)
    if just_price:
        return float(rt['LastTradePrice'])
    return rt


def dask_cov(m, rowvar=1, bias=0, ddof=None):
    """
    Estimate a covariance matrix, given data.
    Covariance indicates the level to which two variables vary together.
    If we examine N-dimensional samples, :math:`X = [x_1, x_2, ... x_N]^T`,
    then the covariance matrix element :math:`C_{ij}` is the covariance of
    :math:`x_i` and :math:`x_j`. The element :math:`C_{ii}` is the variance
    of :math:`x_i`.
    Parameters
    ----------
    m : array_like
        A 1-D or 2-D array containing multiple variables and observations.
        Each row of `m` represents a variable, and each column a single
        observation of all those variables. Also see `rowvar` below.
    y : array_like, optional
        An additional set of variables and observations. `y` has the same
        form as that of `m`.
    rowvar : int, optional
        If `rowvar` is non-zero (default), then each row represents a
        variable, with observations in the columns. Otherwise, the relationship
        is transposed: each column represents a variable, while the rows
        contain observations.
    bias : int, optional
        Default normalization is by ``(N - 1)``, where ``N`` is the number of
        observations given (unbiased estimate). If `bias` is 1, then
        normalization is by ``N``. These values can be overridden by using
        the keyword ``ddof`` in numpy versions >= 1.5.
    ddof : int, optional
        .. versionadded:: 1.5
        If not ``None`` normalization is by ``(N - ddof)``, where ``N`` is
        the number of observations; this overrides the value implied by
        ``bias``. The default value is ``None``.
    Returns
    -------
    out : ndarray
        The covariance matrix of the variables.
    """
    # Check inputs
    if ddof is not None and ddof != int(ddof):
        raise ValueError(
            "ddof must be integer")

    X = m

    if X.shape[0] == 1:
        rowvar = 1
    if rowvar:
        N = X.shape[1]
        axis = 0
    else:
        N = X.shape[0]
        axis = 1

    # check ddof
    if ddof is None:
        if bias == 0:
            ddof = 1
        else:
            ddof = 0
    fact = float(N - ddof)
    if fact <= 0:
        fact = 0.0

    X -= X.mean(axis=1 - axis, keepdims=True)
    if not rowvar:
        return X.T.dot(X) / fact
    else:
        return X.dot(X.T) / fact

from numpy import *


def plot_class(data, axis=(0, 1), axis_names=("Axis1", "Axis2", "Axis3")):
    """Use METHOD enum to choose method for coloring data.
    Set axis to a an array of the axis numbers (starting from 0) or to 'random'
        You can set the axis variable with either 2 or 3 axes
    This method can only be used if your data contains at least 2 axes
    setting usePCA to true is only necessary when your data has more axes than your desired graph type"""

    from matplotlib import pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    colors = array([x for x in 'bgrcmykbgrcmykbgrcmykbgrcmyk'])
    colors = hstack([colors] * 20)

    dim = len(axis)
    if dim > 3 or dim < 2:
        print('Number of axes must be either 2 or 3')
        return

    axes = axis
    # if axis is 'random':
    #     axes = random.permutation(array(range(len(self.__XData[0]))))[:dim]

    x_axis = data[:, axes[0]]
    y_axis = data[:, axes[1]]
    if dim == 3:
        z_axis = data[:, axes[2]]

    x_min = min(x_axis)
    x_max = max(x_axis)
    y_min = min(y_axis)
    y_max = max(y_axis)
    if dim == 3:
        z_min = min(z_axis)
        z_max = max(z_axis)

    fig = plt.figure()

    if dim == 2:
        plt.scatter(x_axis, y_axis, s=15)

        # if method is METHOD.Cluster:
        # if hasattr(self.__model, 'cluster_centers_'):
        #        centers = self.__model.cluster_centers_
        #        if usePCA:
        #            centers = decomposition.PCA(dim).transform(centers)
        #        center_colors = colors[:len(centers)]
        #        plt.scatter(centers[:, 0], centers[:, 1], s=100, c=center_colors)
        plt.xlim(x_min - (0.1 * x_min), x_max + (0.1 * x_max))
        plt.ylim(y_min - (0.1 * y_min), y_max + (0.1 * y_max))
        plt.xticks(())
        plt.yticks(())
        plt.xlabel(axis_names[0])
        plt.ylabel(axis_names[1])

    else:
        ax = Axes3D(fig)
        ax.scatter(x_axis, y_axis, z_axis, cmap=plt.cm.Paired)
        # if method is METHOD.Cluster:
        # if hasattr(self.__model, 'cluster_centers_'):
        #        centers = self.__model.cluster_centers_
        #        if usePCA:
        #            centers = decomposition.PCA(dim).transform(centers)
        #        center_colors = colors[:len(centers)]
        #        ax.scatter(centers[:, axes[0]], centers[:, axes[1]], centers[:, axes[2]], c=center_colors, s=100)

        ax.set_xlabel(axis_names[0])
        ax.w_xaxis.set_ticklabels([])
        ax.set_ylabel(axis_names[1])
        ax.w_yaxis.set_ticklabels([])
        ax.set_zlabel(axis_names[2])
        ax.w_zaxis.set_ticklabels([])
        ax.autoscale_view()
        ax.autoscale()
    plt.show()
