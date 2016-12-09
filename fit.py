#!/usr/local/bin/python
import csv
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures

from panda.debug import debug


# we want 1 equation to fit both curves using few coefficients,
# obviously coefficients willl be different for both data sets,
# I have a 3rd data set that we can look at if need be and maybe have some others around

# https://docs.scipy.org/doc/numpy-1.10.0/reference/generated/numpy.polyfit.html


def get_data(fname):
    with open(fname) as f:
        reader = csv.DictReader(f)
        d = [r for r in reader]

    x = np.array([float(r['sintl2']) for r in d])
    y = np.array([float(r['I1/I2%']) for r in d])

    return x, y


def compute_and_plot_fits_np(x, y):
    xi = np.linspace(min(x), max(x), 100)
    yi1 = np.polyval(np.polyfit(x, y, 1), xi)
    yi2 = np.polyval(np.polyfit(x, y, 2), xi)
    yi3 = np.polyval(np.polyfit(x, y, 3), xi)
    plt.plot(x, y, 'k.', label='data')
    plt.plot(xi, yi1, 'b-', label='linear')
    plt.plot(xi, yi2, 'r-', label='quadratic')
    plt.plot(xi, yi3, 'g-', label='cubic')
    plt.legend()


def compute_and_plot_fits(x, y, name):
    # similar to the np interface, but more options available, like ransac
    x_ = x.reshape((-1, 1))
    y_ = y.reshape((-1, 1))

    xi = np.linspace(min(x), max(x), 100).reshape((-1, 1))
    yi = []

    poly_2 = PolynomialFeatures(degree=2)
    x_2 = poly_2.fit_transform(x_)
    xi_2 = poly_2.fit_transform(xi)

    m = linear_model.LinearRegression()
    m.fit(x_, y_)
    yi.append(m.predict(xi))

    m = linear_model.RANSACRegressor(linear_model.LinearRegression())
    m.fit(x_, y_)
    yi.append(m.predict(xi))

    m = linear_model.LinearRegression()
    m.fit(x_2, y_)
    yi.append(m.predict(xi_2))

    m = linear_model.RANSACRegressor(linear_model.LinearRegression())
    m.fit(x_2, y_)
    yi.append(m.predict(xi_2))

    plt.plot(x, y, 'k.', label='data')
    plt.plot(xi, yi[0], label='linear')
    plt.plot(xi, yi[1], label='linear ransac')
    plt.plot(xi, yi[2], label='quadratic')
    plt.plot(xi, yi[3], label='quadratic ransac')
    plt.title(name)
    plt.legend()


def quadratic_ransac_fit(x, y, name):
    # need to use a constrained parabola OR a decaying exponential
    # RANSACRegressor takes a model that implements fit(X, y) and score(X, y)
    # need to define one of those models
    # - exponential - just fit (x, log(y))
    # - constrained parabola - from scipy.optimize import curve_fit
    #   would probably have to implement ransac myself for that to work...

    transform_func, transform_ifunc = np.log, np.exp
    # transform_func, transform_ifunc = lambda x: x, lambda x: x

    y_log = transform_func(y)

    x_ = x.reshape((-1, 1))
    y_ = y_log.reshape((-1, 1))

    xi = np.linspace(min(x), max(x), 100).reshape((-1, 1))

    poly_2 = PolynomialFeatures(degree=2)
    x_2 = poly_2.fit_transform(x_)
    xi_2 = poly_2.fit_transform(xi)

    m = linear_model.RANSACRegressor(linear_model.LinearRegression())
    m.fit(x_2, y_)
    yi_log = m.predict(xi_2)
    c = m.estimator_.coef_
    yi_log_b = np.dot(c, xi_2.T).T

    # the coefficients dont include the x^0 term for some reason??
    c_b = np.array([float(yi_log[3][0] - yi_log_b[0]), c[0, 1], c[0, 2]])
    yi_log_b = np.dot(c_b, xi_2.T).T

    yi = transform_ifunc(yi_log)
    yi_b = transform_ifunc(yi_log_b)

    inlier_mask = m.inlier_mask_
    outlier_mask = np.logical_not(inlier_mask)

    plt.plot(x[inlier_mask], y[inlier_mask], 'k.', label='data')
    plt.plot(x[outlier_mask], y[outlier_mask], 'r.', label='data (outliers)')
    plt.plot(xi, yi, label='quadratic ransac')
    # plt.plot(xi, yi_b, 'r--')
    plt.title('%s: %0.5f + %0.5fx + %0.5fx^2' % (name, c_b[0], c_b[1], c_b[2]))
    plt.legend()


if __name__ == '__main__':
    names = ['clq', 'uf6']

    n = 1
    for name in names:
        x, y = get_data(name + '.csv')
        plt.subplot(2, 1, n)
        quadratic_ransac_fit(x, y, name)
        n += 1

    plt.show()
    debug()
