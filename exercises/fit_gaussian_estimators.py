import numpy.random
import plotly.offline

from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

import plotly.express as px

pio.templates.default = "simple_white"


def test_univariate_gaussian():
    # Question 1 - Draw samples and print fitted model
    uni = UnivariateGaussian()
    samples = np.random.normal(10, 1, 1000)

    uni.fit(samples)
    print(f"({uni.mu_}, {uni.var_})")

    # Question 2 - Empirically showing sample mean is consistent
    expectations = np.zeros(100)
    for i in range(1, 101):
        uni.fit(samples[:i * 10])
        expectations[i - 1] = np.abs(uni.mu_ - 10)

    # TODO: GO: plotly.graph_objects
    figure = px.line(x=[i * 10 for i in range(1, 101)], y=expectations,
                     labels=dict(x="Sample size", y="Distance from estimated to true expectation"))
    # figure.show()
    plotly.offline.plot(figure, filename="temp-plot-1.html")

    # Question 3 - Plotting Empirical PDF of fitted model
    # last fit was 1000: same as Q1
    figure2 = px.scatter(x=samples, y=uni.pdf(samples),
                         labels=dict(x="Value of sample", y="PDF of sample"))

    plotly.offline.plot(figure2)
    # figure2.show()


def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    mu = np.array([0, 0, 4, 0])
    sigma = np.array([[1, 0.2, 0, 0.5],
                      [0.2, 2, 0, 0],
                      [0, 0, 1, 0],
                      [0.5, 0, 0, 1]])

    multi = MultivariateGaussian()
    samples = numpy.random.multivariate_normal(mu, sigma, 1000)
    multi.fit(samples)
    print(multi.mu_)
    print(multi.cov_)

    # Question 5 - Likelihood evaluation
    raise NotImplementedError()

    # Question 6 - Maximum likelihood
    raise NotImplementedError()


if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()
