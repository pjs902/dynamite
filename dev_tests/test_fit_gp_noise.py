"""fit_gp must recover a known injected noise level, at any n_train.

Regression guard for the old n_train>300 switch to SingleTaskVariationalGP,
which was fit by ELBO against unstandardized chi2 and silently returned a
flat posterior with all the variance in the likelihood noise (26187 recovered
against an injected sigma of 50). Nothing raised -- only a calibration check
like this one catches it.

NOTE: run against the working tree, not the installed package --
    PYTHONPATH=$(git rev-parse --show-toplevel) python dev_tests/test_fit_gp_noise.py
"""
import warnings

import numpy as np
import torch

from dynamite.parameter_space import fit_gp

SIGMA = 50.0


def noise_std(model, X):
    """Likelihood noise in chi2 units: sqrt(mean(Var[y] - Var[f]))."""
    X_t = torch.tensor(np.atleast_2d(X), dtype=torch.double)
    with torch.no_grad():
        var_y = model.posterior(X_t, observation_noise=True).variance
        var_f = model.posterior(X_t).variance
    excess = (var_y - var_f).clamp_min(0.0).numpy().ravel()
    return float(np.sqrt(np.mean(excess)))


def landscape(n, seed=0):
    """Smooth quadratic in chi2 units of order 3e5, plus sigma=SIGMA scatter."""
    rng = np.random.default_rng(seed)
    X = rng.random((n, 3))
    chi2 = 3e5 + 2e4 * ((X - 0.4) ** 2).sum(1) + rng.normal(0, SIGMA, n)
    return X, chi2


def main():
    warnings.filterwarnings("ignore")
    for n in (290, 310, 600):
        X, chi2 = landscape(n)
        model = fit_gp(X, chi2)
        sigma_hat = noise_std(model, X)
        print(f"n={n:4d}  {type(model).__name__:14s}  sigma_hat={sigma_hat:8.1f}")
        # factor 2 is loose on purpose: the point is orders of magnitude.
        assert 0.5 * SIGMA < sigma_hat < 2 * SIGMA, (
            f"n={n}: recovered noise floor {sigma_hat:.0f} vs injected {SIGMA}"
        )
    print("ok")


if __name__ == "__main__":
    main()
