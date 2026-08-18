import os
import numpy as np
import copy

import torch
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.optim import optimize_acqf
from botorch.models.transforms.outcome import Standardize
from botorch.models.transforms.input import Normalize, Round, ChainedInputTransform
from botorch.acquisition.monte_carlo import qUpperConfidenceBound
from botorch.acquisition.logei import qLogExpectedImprovement
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils.rounding import approximate_round


from samplers.base import Sampler


class SOBOSampler(Sampler):
    def __init__(self, config):
        super().__init__(config)
        self.acquisition = config['bo']['acquisition']


    def ask(self, n = 1):

        train_x = torch.from_numpy(self.train_x)
        train_y = torch.from_numpy(self.train_y).unsqueeze(-1) # shape: (BOiter, 1)
        bounds = torch.from_numpy(self.bounds)

        assert self.train_x.shape[0] > 0

        # Optimizing in [0, 1) unit cube is standard for BO; also numerically more stable.
        # See also acqf_bounds
        train_x_normalized = (train_x - bounds[0]) / (bounds[1] - bounds[0])

        input_transform = ChainedInputTransform(
            unnormalize = Normalize(
                d=train_x.shape[1],
                bounds=bounds,
                transform_on_train=True, transform_on_eval=True,
                reverse=True
            ),
            round = Round(
                integer_indices=self.integer_indices,
                transform_on_train=True, transform_on_eval=True,
                approximate=True, tau=1e-3,
            ),
            normalize = Normalize(
                d=train_x.shape[1],
                bounds=bounds,
                transform_on_train=True, transform_on_eval=True
            )
        )

        outcome_transform = Standardize(m=1, min_stdv=1e-8)

        gp = SingleTaskGP(
            train_x_normalized,
            train_y,
            input_transform=input_transform,
            outcome_transform=outcome_transform
        )
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_mll(mll)


        sampler = SobolQMCNormalSampler(sample_shape=torch.Size([512]))

        if self.acquisition == 'ucb':
            acqf = qUpperConfidenceBound(gp, beta=self.config['bo']['beta'], sampler=sampler)
        elif self.acquisition == 'ei':
            acqf = qLogExpectedImprovement(gp, best_f=train_y.max(), sampler=sampler)
        else:
            raise NotImplementedError(f"Acquisition function {self.acquisition} is not implemented. Current options: 'ucb', 'ei'")
    
        acqf_bounds = torch.stack([
            torch.zeros(train_x.shape[1], dtype=torch.double),
            torch.ones(train_x.shape[1], dtype=torch.double),
        ])

        candidates, _ = optimize_acqf(
            acq_function=acqf,
            bounds=acqf_bounds,
            q=n,
            num_restarts=20,
            raw_samples=1024,
            post_processing_func=self._pr_post_processing,  # PR applied here
            sequential=True,
        )

        new_xs = candidates.detach() * (bounds[1] - bounds[0]) + bounds[0]

        # Hard-round integer dims (final guarantee)
        for i in self.integer_indices:
            new_xs[:, i] = torch.round(new_xs[:, i])

        next_configs = []
        for i in range(n):
            next_config = copy.deepcopy(self.config)
            for j, param in enumerate(self.params):
                next_config['ptycho']['params'][param] = new_xs[i,j].item()
            next_configs.append(next_config)
        
        return next_configs


    def _pr_post_processing(self, X):
        """Apply differentiable rounding to integer dims (PR forward pass)."""
        X_out = X.clone()
        for idx in self.integer_indices:
            # Unnormalize -> approximate_round -> renormalize
            raw = X_out[..., idx] * (self.bounds[1][idx] - self.bounds[0][idx]) + self.bounds[0][idx]
            rounded = approximate_round(raw)
            X_out[..., idx] = (rounded - self.bounds[0][idx]) / (self.bounds[1][idx] - self.bounds[0][idx])
        return X_out


    





