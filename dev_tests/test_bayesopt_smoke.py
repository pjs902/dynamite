"""Environment smoke test for the BayesOpt stack: torch + botorch + gpytorch.

No DYNAMITE imports. Run on any node before production:
    python dev_tests/test_bayesopt_smoke.py
"""

import numpy as np
import torch
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import qLogExpectedImprovement
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood


def main():
    torch.manual_seed(0)
    d = 3
    X = torch.rand(20, d, dtype=torch.double)
    y = -((X - 0.5) ** 2).sum(dim=-1, keepdim=True) * 10.0
    gp = SingleTaskGP(X, y).to(torch.double)
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    fit_gpytorch_mll(mll)
    acqf = qLogExpectedImprovement(gp, best_f=y.max())
    bounds = torch.stack([torch.zeros(d), torch.ones(d)]).to(torch.double)
    cands, acq_val = optimize_acqf(acq_function=acqf, bounds=bounds, q=2, num_restarts=2, raw_samples=16)
    assert cands.shape == (2, d), cands.shape
    assert torch.isfinite(cands).all() and torch.isfinite(acq_val)
    print(f"SMOKE OK: acq_value={acq_val.item():.4f} candidates={cands.squeeze().tolist()}")


if __name__ == "__main__":
    main()
