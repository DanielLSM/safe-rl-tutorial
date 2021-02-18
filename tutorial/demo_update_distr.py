# https://stats.stackexchange.com/questions/237037/bayesian-updating-with-new-data
# https://www.cs.ubc.ca/~murphyk/Papers/bayesGauss.pdf

import numpy as np
from matplotlib import pyplot as plt

from conjugate_prior import NormalNormalKnownVar
model = NormalNormalKnownVar(1)
model.plot(-5, 5)
plt.show()
new_model = model

for _ in range(1000):
    new_model = NormalNormalKnownVar(0.01,
                                     prior_mean=(new_model.mean + 0.05),
                                     prior_var=0.01)
    model = model.update([new_model.sample()])
    model.plot(-5, 5)
print(model.sample())
plt.show()
# new_model = new_model.update([0.1])

# heads = 95
# tails = 105
# prior_model = BetaBinomial()  # Uninformative prior
# updated_model = prior_model.update(heads, tails)
# credible_interval = updated_model.posterior(0.45, 0.55)
# print("There's {p:.2f}% chance that the coin is fair".format(
#     p=credible_interval * 100))
# predictive = updated_model.predict(50, 50)
# print("The chance of flipping 50 Heads and 50 Tails in 100 trials is {p:.2f}%".
#       format(p=predictive * 100))

# def bern_post(n_params=100, n_sample=100, true_p=.8, prior_p=.5, n_prior=100):
#     params = np.linspace(0, 1, n_params)
#     sample = np.random.binomial(n=1, p=true_p, size=n_sample)
#     likelihood = np.array(
#         [np.product(st.bernoulli.pmf(sample, p)) for p in params])
#     #likelihood = likelihood / np.sum(likelihood)
#     prior_sample = np.random.binomial(n=1, p=prior_p, size=n_prior)
#     prior = np.array(
#         [np.product(st.bernoulli.pmf(prior_sample, p)) for p in params])
#     prior = prior / np.sum(prior)
#     posterior = [prior[i] * likelihood[i] for i in range(prior.shape[0])]
#     posterior = posterior / np.sum(posterior)

#     fig, axes = plt.subplots(3, 1, sharex=True, figsize=(8, 8))
#     axes[0].plot(params, likelihood)
#     axes[0].set_title("Sampling Distribution")
#     axes[1].plot(params, prior)
#     axes[1].set_title("Prior Distribution")
#     axes[2].plot(params, posterior)
#     axes[2].set_title("Posterior Distribution")
#     sns.despine()
#     plt.tight_layout()

#     return posterior

# example_post = bern_post()