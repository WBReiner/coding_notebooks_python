#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 26 10:23:45 2018

@author: whitneyreiner
"""

# =============================================================================
# Histograms
# From: https://realpython.com/python-histograms/
# histograms: tool for quickly assessing a probability distribution. 
# https://github.com/realpython/materials/blob/master/histograms/histograms.py
# =============================================================================
#%%
# =============================================================================
# Need not be sorted, necessarily
a = (0, 1, 1, 1, 2, 3, 7, 7, 23)

#make a dictionary:
def count_elements(seq) -> dict:
# Tally elements from `seq`.
    hist = {}
    for i in seq:
        hist[i] = hist.get(i, 0) + 1 # “for each element of the sequence, 
        # increment its corresponding value in hist by 1.”
    return hist

counted = count_elements(a)
counted
#%%
# =============================================================================
# In fact, this is precisely what is done by the collections.Counter class from 
# Python’s standard library, which subclasses a Python dictionary and overrides 
# its .update() method:
# =============================================================================
#%%
# Can also use counter
from collections import Counter

recounted = Counter(a)
recounted
#test the two are equal
recounted.items() == counted.items()

        
import random
random.seed(1)
#%%
 =============================================================================
# Thus far, you have been working with what could best be called 
# “frequency tables.” But mathematically, a histogram is a mapping of bins 
# (intervals) to frequencies. More technically, it can be used to approximate the
#  probability density function (PDF) of the underlying variable.
# =============================================================================
#%%
# =============================================================================
# Technical Detail: All but the last (rightmost) bin is half-open. That is, all 
# bins but the last are [inclusive, exclusive), and the final bin is
#     [inclusive, inclusive].
# =============================================================================
#%%
# =============================================================================
# Staying in Python’s scientific stack, Pandas’ Series.histogram() uses 
# matplotlib.pyplot.hist() to draw a Matplotlib histogram of the input Series:
# =============================================================================
import pandas as pd
import numpy as np
# Generate data on commute times.
size, scale = 1000, 10
commutes = pd.Series(np.random.gamma(scale, size=size) ** 1.5)

commutes.plot.hist(grid=True, bins=20, rwidth=0.9,
                   color='#607c8e')
plt.title('Commute Times for 1,000 Commuters')
plt.xlabel('Counts')
plt.ylabel('Commute Time')
plt.grid(axis='y', alpha=0.75)
#%%
# =============================================================================
# pandas.DataFrame.histogram() is similar but produces a histogram for each 
# column of data in the DataFrame.
# =============================================================================
#%%
# =============================================================================
# A kernel density estimation (KDE) is a way to estimate the probability density 
# function (PDF) of the random variable that “underlies” our sample. KDE is a
#  means of data smoothing.
# =============================================================================
#%%
# Sample from two different normal distributions
means = 10, 20
stdevs = 4, 2
dist = pd.DataFrame(np.random.normal(loc=means, scale=stdevs, size=(1000, 2)),
                    columns=['a', 'b'])
dist.agg(['min', 'max', 'mean', 'std']).round(decimals=2)
#%%
#Now, to plot each histogram on the same Matplotlib axes:
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
dist.plot.kde(ax=ax, legend=False, title='Histogram: A vs. B')
dist.plot.hist(density=True, ax=ax)
ax.set_ylabel('Probability')
ax.grid(axis='y')
ax.set_facecolor('#d8dcd6')
#%%
# =============================================================================
# If you take a closer look at this function, you can see how well it approximates the “true” PDF for a relatively small sample of 1000 data points. Below, you can first build the “analytical” distribution with scipy.stats.norm(). This is a class instance that encapsulates the statistical standard normal distribution, its moments, and descriptive functions. Its PDF is “exact” in the sense that it is defined precisely as norm.pdf(x) = exp(-x**2/2) / sqrt(2*pi).
# 
# Building from there, you can take a random sample of 1000 datapoints from this distribution, then attempt to back into an estimation of the PDF with scipy.stats.gaussian_kde():
# 
# =============================================================================
from scipy import stats

# An object representing the "frozen" analytical distribution
# Defaults to the standard normal distribution, N~(0, 1)
dist = stats.norm()

# Draw random samples from the population you built above.
# This is just a sample, so the mean and std. deviation should
# be close to (1, 0).
samp = dist.rvs(size=1000)

# `ppf()`: percent point function (inverse of cdf — percentiles).
x = np.linspace(start=stats.norm.ppf(0.01),
                stop=stats.norm.ppf(0.99), num=250)
gkde = stats.gaussian_kde(dataset=samp)

# `gkde.evaluate()` estimates the PDF itself.
fig, ax = plt.subplots()
ax.plot(x, dist.pdf(x), linestyle='solid', c='red', lw=3,
        alpha=0.8, label='Analytical (True) PDF')
ax.plot(x, gkde.evaluate(x), linestyle='dashed', c='black', lw=2,
        label='PDF Estimated via KDE')
ax.legend(loc='best', frameon=False)
ax.set_title('Analytical vs. Estimated PDF')
ax.set_ylabel('Probability')
ax.text(-2., 0.35, r'$f(x) = \frac{\exp(-x^2/2)}{\sqrt{2*\pi}}$',
        fontsize=12)
#%%
# =============================================================================
# This is a bigger chunk of code, so let’s take a second to touch on a few key lines:
# 
# SciPy’s stats subpackage lets you create Python objects that represent analytical distributions that you can sample from to create actual data. So dist = stats.norm() represents a normal continuous random variable, and you generate random numbers from it with dist.rvs().
# To evaluate both the analytical PDF and the Gaussian KDE, you need an array x of quantiles (standard deviations above/below the mean, for a normal distribution). stats.gaussian_kde() represents an estimated PDF that you need to evaluate on an array to produce something visually meaningful in this case.
# The last line contains some LaTex, which integrates nicely with Matplotlib.
# =============================================================================
#%%
# =============================================================================
# A Fancy Alternative with Seaborn
# Let’s bring one more Python package into the mix. Seaborn has a displot() function that plots the histogram and KDE for a univariate distribution in one step. Using the NumPy array d from ealier:
# 
# =============================================================================
import seaborn as sns

import numpy as np
# `numpy.random` uses its own PRNG.
np.random.seed(444)
np.set_printoptions(precision=3)

d = np.random.laplace(loc=15, scale=3, size=500)
d[:5]
sns.set_style('darkgrid')
sns.distplot(d)
#%%
# =============================================================================
# The call above produces a KDE. There is also optionality to fit a specific distribution to the data. This is different than a KDE and consists of parameter estimation for generic data and a specified distribution name:
# =============================================================================
sns.distplot(d, fit=stats.laplace, kde=False)
# =============================================================================
# Again, note the slight difference. In the first case, you’re estimating some unknown PDF; in the second, you’re taking a known distribution and finding what parameters best describe it given the empirical data.
# =============================================================================
#%%
# =============================================================================
# In addition to its plotting tools, Pandas also offers a convenient .value_counts() method that computes a histogram of non-null values to a Pandas Series:
# 
# =============================================================================
import pandas as pd

data = np.random.choice(np.arange(10), size=10000, p=np.linspace(1, 11, 10) / 60)
s = pd.Series(data)

s.value_counts()
#%%
s.value_counts(normalize=True).head()
#%%
# =============================================================================
# Elsewhere, pandas.cut() is a convenient way to bin values into arbitrary intervals. Let’s say you have some data on ages of individuals and want to bucket them sensibly:
# =============================================================================
ages = pd.Series([1, 1, 3, 5, 8, 10, 12, 15, 18, 18, 19, 20, 25, 30, 40, 51, 52])
bins = (0, 10, 13, 18, 21, np.inf)  # The edges
labels = ('child', 'preteen', 'teen', 'military_age', 'adult')
groups = pd.cut(ages, bins=bins, labels=labels)

groups.value_counts()
#%%
pd.concat((ages, groups), axis=1).rename(columns={0: 'age', 1: 'group'})
#%%
  