#!/usr/bin/env python

from __future__ import print_function, division
import numpy as np
from scipy.stats import norm, multivariate_normal
from scipy.optimize import minimize
from scipy.misc import logsumexp
from tqdm import tqdm
from copy import deepcopy
from pymc import ZeroProbability


def sample(model, n, n_burn=0, beta=0.2, tune_beta=False, eye=False):
    """
    This sampler implements Bayesian annealed sequential importance sampling (BASIS), as described by Wu2017. In
    particular, this function implements the special case with a maximum chain length of one (l_max = 1). We combine
    the algorithm by Wu2017 with an optional tuning method for the proposal scaling factor beta, as described by
    Minson2013. The sample function can perform inference on any valid pymc-model object.

    Args:
        model: PyMC model instance (see https://pymc-devs.github.io/pymc/index.html)
        n: Number of samples to draw (i.e. number of parallel Markov chains to run)
        n_burn: Number of burn-in samples for each Markov chain
        beta: Scaling factor of proposal PDF (default value 0.2 taken from Ching2007)
        tune_beta: Activates adaptive tuning of beta, as described in Minson2013
        eye: If true, proposal PDF neglects covariance of parameters

    Returns:
        dict: Dictionary with parameter names as keys and corresponding samples as values
        float: Log10-value of the marginal likelihood (model evidence)

    References:
        Ching2007: https://doi.org/10.1061/(ASCE)0733-9399(2007)133:7(816)
        Minson2013: https://doi.org/10.1093/gji/ggt180
        Wu2017: https://doi.org/10.1115/1.4037450
    """
    # Determine which model variables are integers
    variables = [var for var in model.variables if not var.observed]
    observed = [var for var in model.variables if var.observed]
    integers = [var.dtype == np.int for var in model.variables if not var.observed]

    # Prior samples
    print('Sampling from prior...')
    prior = []
    for i in range(n):
        model.draw_from_prior()
        prior.append([var.value for var in variables])

    prior = np.array(prior)
    print('Entering annealing loop...')
    print('')

    # Main loop
    p = 0
    dp = 0.1

    s = []
    posterior = deepcopy(prior)

    epoch = 1

    while p < 1:
        # log-likelihood for all samples
        llh = []
        for sample in posterior:
            # set values in pymc model to evaluate log-probability
            for i, var in enumerate(variables):
                var.value = sample[i]

            llh.append(np.sum([var.logp for var in observed]))

        llh = np.array(llh)

        # p is increased by dp so that the coefficient of variation (std/mean) of the weights is (approx.) one; we do
        # this by minimizing (std/mean - 1)**2.; optimization is done in log-space as order of magnitude of dp depends
        # on number of data points and model structure
        log_dp = np.log(dp)
        res = minimize(lambda x: _minfunc(x, llh), log_dp, method='COBYLA', options={'maxiter': 1000, 'rhobeg': 1})
        dp = np.exp(res.x)

        # if dp increases p to be larger than one, we reset dp so that p is set to one
        if p + dp > 1:
            dp = 1 - p
        p += dp

        print('Epoch {} --- p = {:.6f}'.format(epoch, p))

        w = np.exp(llh * dp)  # rescaled plausibility weights
        w_normed = w / np.sum(w)
        s.append(np.log10(np.mean(w)))  # marginal likelihood factor

        # resampling
        print('Resampling...')
        indices = np.random.choice(np.arange(n), size=n, p=w_normed)
        posterior = posterior[indices]

        # MCMC
        print('Metropolis sampling...')
        if eye:
            sigma = np.std(posterior, axis=0)
            proposal = norm(loc=[0] * len(sigma), scale=(beta ** 2) * sigma)
        else:
            cov = np.cov(posterior, rowvar=False)
            if cov.ndim == 0:  # check if only one parameter
                cov = np.array([cov])

            proposal = multivariate_normal(mean=[0] * len(cov), cov=(beta ** 2) * cov, allow_singular=True)

        with tqdm(total=(n_burn+1)*n) as pbar:  # initialize progressbar
            for i in range((n_burn+1)):
                # bookkeeping variables for determining the acceptance rate
                n_candidates = 0
                n_accept = 0

                for j in range(n):
                    n_candidates += 1

                    d = proposal.rvs()  # draw from proposal distribution

                    if d.ndim == 0:  # check if only one parameter
                        d = np.array([d])

                    d[integers] = np.round(d[integers])  # proposal is rounded to nearest integer for integer variables

                    candidate = posterior[j] + d  # create candidate sample

                    # set values in pymc model to evaluate log-probability
                    for k, var in enumerate(variables):
                        var.value = candidate[k]

                    # Apply Metropolis algorithm for accepting/declining candidate sample; pymc will raise ane error if
                    # the candidate sample lies outside the variable's support, we catch this error and do not update
                    # the variable's value in this case (as the probability of this value is zero)
                    try:
                        # model logp needs to be evaluated to raise error if any variable
                        # is outside of its respective support region
                        model.logp

                        new_llh = np.sum([var.logp for var in observed])

                        alpha = new_llh - llh[j]
                        u = np.log(np.random.uniform(0, 1))

                        if u <= alpha:
                            posterior[j] = candidate
                            n_accept += 1
                    except ZeroProbability:
                        pass

                    pbar.update(1)  # update progressbar

        # Compute acceptance rate of last round of sampling
        accept_rate = float(n_accept)/n_candidates

        # Tuning method of the proposal scaling from Minson2013
        if tune_beta:
            beta = (1./9.) + accept_rate*(8./9.)
            print('Acceptance rate: {:.2f}% (new beta={:.3f})'.format(accept_rate * 100.0, beta))
        else:
            print('Acceptance rate: {:.2f}%'.format(accept_rate * 100.0))

        epoch += 1
        print('')

    marglike = np.sum(s)  # log10-value of marginal likelihood (model evidence)

    # Create dictionary with variable names and corresponding samples
    names = [str(var) for var in variables]
    samples = {name: values for name, values in zip(names, posterior.T)}

    print('Done. Marginal likelihood: 10^{:.3f}'.format(marglike))
    return samples, marglike


def _minfunc(log_dp, llh):
    """
    Helper function for the optimization routine. It computes plausibility weights for a given proposed step size of p
    and returns the quadratic deviation of the coefficient of variation of those weights from the target value 1.

    Args:
        log_dp(float): Proposed step size for annealing parameter p.
        llh: Current log-likelihood values.

    Returns:
        float: Quadratic deviation of the coefficient of variation from the target value 1.
    """
    # Coefficient of variation (numerically stable)
    lw = llh * np.exp(log_dp)
    lw_mean = logsumexp(lw) - np.log(len(lw))

    var = np.std(np.exp(lw - lw_mean))  # computes np.std(w) / np.mean(w) in log-space
    return (var - 1) ** 2.
