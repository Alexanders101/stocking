# coding: utf-8

import sys; print('Python %s on %s' % (sys.version, sys.platform))
sys.path.extend(['/home/alex/PycharmProjects/stock'])
from utils import get_stock
get_stock("YHOO")
import datetime
datetime.date(2014, 1, 1)
start = datetime.date(2014, 1, 1)
yahoo = get_stock('YHOO', start)
yahoo
yahoo.Open
yahoo.Open[:-1]
yhoo = yahoo.Open[:-1]
yhoo
import statsmodels
ar = statsmodels.tsa.ar_model.AR(yhoo)
from statsmodels import tsa
from statsmodels.tsa
from statsmodels.tsa.ar_model import AR
ar = AR(yhoo)
ar
ar.fit()
results = ar.fit()
reults
results
print(results)
results.predict()
yhoo
yhoo - results.predict()
yhoo - results.predict().mean()
yhoo - results.predict().mean(0)
yhoo - results.predict().mean(1)
yhoo - results.predict().mean()
(yhoo - results.predict()),mean()
(yhoo - results.predict()).mean()
(yhoo - results.predict()).std()
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
range = np.arange(-10, 10, 0.001)
plt.plot(range, norm.pdf(range,(yhoo - results.predict()).mean(), (yhoo - results.predict()).std()))
get_ipython().magic(u'save ar')
get_ipython().magic(u"save 'ar'")
get_ipython().magic(u'save')
