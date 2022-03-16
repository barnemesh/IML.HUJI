import numpy.random

from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import plotly.express as px

# import plotly.offline
# import time

pio.templates.default = "simple_white"


def test_univariate_gaussian():
    # Question 1 - Draw samples and print fitted model
    uni = UnivariateGaussian()
    samples = np.random.normal(10, 1, 1000)

    uni.fit(samples)
    print("(expectation, variance)")
    print(f"({uni.mu_}, {uni.var_})")

    # Question 2 - Empirically showing sample mean is consistent
    expectations = np.zeros(100)
    for i in range(1, 101):
        uni.fit(samples[:i * 10])
        expectations[i - 1] = np.abs(uni.mu_ - 10)

    figure = px.line(x=[i * 10 for i in range(1, 101)], y=expectations,
                     title="Distance from estimated to true expectation based on sample size",
                     labels=dict(x="Sample size",
                                 y="Distance from estimated to true expectation"))
    figure.show()

    # Question 3 - Plotting Empirical PDF of fitted model
    # last fit was 1000: same as Q1
    figure2 = px.scatter(x=samples, y=uni.pdf(samples),
                         title="Calculated PDF of the samples in Univariate gaussian estimator",
                         labels=dict(x="Value of sample", y="PDF of sample"))

    figure2.show()


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
    print("Estimated expectation")
    print(multi.mu_)
    print("Unbiased estimated Covariance matrix")
    print(multi.cov_)

    # Question 5 - Likelihood evaluation
    values_for_mu = np.linspace(-10, 10, 200)
    rep = np.repeat(values_for_mu, len(values_for_mu))
    pairs_f1_f3 = np.transpose(
        np.array([np.repeat(values_for_mu, len(values_for_mu)),
                  np.zeros(len(rep)),
                  np.tile(values_for_mu, len(values_for_mu)),
                  np.zeros(len(rep))
                  ])
    )

    def multi_loglikely(m):
        return MultivariateGaussian.log_likelihood(m, sigma, samples)

    # start = time.time()
    # print("hello")
    log_likelihoods = np.apply_along_axis(multi_loglikely, 1, pairs_f1_f3)
    # end = time.time()
    # print(end - start)

    fig = go.Figure(
        go.Heatmap(y=pairs_f1_f3[:, 0],
                   x=pairs_f1_f3[:, 2],
                   z=log_likelihoods,
                   colorbar={"title": "Value of Log-Likelihood"}
                   )
    )
    fig.update_layout(
        title="Log-likelihood using mu=[f1,0,f3,0]^T as expectation, sampled from [-10,10]",
        yaxis_title="value of f1",
        xaxis_title="value of f3",
    )
    fig.show()
    # plotly.offline.plot(fig, filename="first.html")

    figure2 = px.density_heatmap(y=pairs_f1_f3[:, 0],
                                 x=pairs_f1_f3[:, 2],
                                 z=log_likelihoods,
                                 labels=dict(y="value of f1", x="value of f3",
                                             z="calculated log-likelihood"),
                                 title="Log-likelihood using mu=[f1,0,f3,0]^T as expectation, sampled from [-10,10]",
                                 histfunc="avg",
                                 histnorm="density"
                                 )
    figure2.show()
    # plotly.offline.plot(figure2, filename="second.html")
    # Question 6 - Maximum likelihood
    mle_index = np.argmax(log_likelihoods)
    print("f1: {:.3f} , f3: {:.3f}".format(
        *pairs_f1_f3[mle_index, [0, 2]])
    )
    print(f"Log-Likelihood: {log_likelihoods[mle_index]:.3f}")


if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()
