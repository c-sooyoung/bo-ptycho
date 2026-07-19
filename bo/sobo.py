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


from bo.bo_base import BOEngine


class SingleObjectiveBOEngine(BOEngine):
    def __init__(self, config):
        super().__init__(config)

        self.params = [key for key, spec in config["bo"]["params"].items() if spec is not None]
        self.param_types = {key: config["bo"]["params"][key].get("type", "float") for key in self.params}
        self.integer_indices = [i for i, param in enumerate(self.params) if self.param_types[param] == 'int']
        self.bounds = np.empty((2, len(self.params)))

        for i, param in enumerate(self.params):
            center = config["ptycho"]["params"][param]
            radius = config["bo"]["params"][param]["radius"]
            self.bounds[0, i] = center - radius
            self.bounds[1, i] = center + radius

        self.train_x = np.empty((0, len(self.params)))  # shape: (BOiter, BOparam)
        self.train_y = np.empty((0,))                   # shape: (BOiter,)

        train_x_path = config["bo"].get("train_x")
        train_y_path = config["bo"].get("train_y")

        if train_x_path is not None and train_y_path is not None:
            if os.path.exists(train_x_path) and os.path.exists(train_y_path):
                train_x = np.load(train_x_path)
                train_y = np.load(train_y_path)
                assert train_x.ndim == 2,                    "loaded train_x must be 2D"
                assert train_x.shape[1] == len(self.params), "loaded train_x shape(1) does not match number of variable parameters"
                assert train_y.ndim == 1,                    "loaded train_y must be 1D"
                assert train_y.shape[0] == train_x.shape[0], "loaded train_x and train_y shape(0) have unequal iterations"
                self.train_x = train_x
                self.train_y = train_y
        
        self.acquisition = config['bo']['acquisition']


    def ask(self):

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
            beta = 0.2
            print("Acquisition: UCB | Beta: {} (fixed)".format(beta))
            acqf = qUpperConfidenceBound(gp, beta=beta, sampler=sampler)
        elif self.acquisition == 'ei':
            best_f = train_y.max()
            print("Acquisition: LogEI  best_f: {:.6f}".format(best_f.item()))
            acqf = qLogExpectedImprovement(gp, best_f=best_f, sampler=sampler)
        else:
            raise NotImplementedError(f"Acquisition function {self.acquisition} is not implemented. Current options: 'ucb', 'ei'")
    
        # Full [0,1]^d search (trust region disabled)
        acqf_bounds = torch.stack([
            torch.zeros(train_x.shape[1], dtype=torch.double),
            torch.ones(train_x.shape[1], dtype=torch.double),
        ])

        candidate, _ = optimize_acqf(
            acq_function=acqf,
            bounds=acqf_bounds,
            q=1,
            num_restarts=20,
            raw_samples=1024,
            post_processing_func=self._pr_post_processing,  # PR applied here
            sequential=True,
        )

        new_x = candidate.detach() * (bounds[1] - bounds[0]) + bounds[0]

        # Hard-round integer dims (final guarantee)
        for i in self.integer_indices:
            new_x[:, i] = torch.round(new_x[:, i])

        next_config = copy.deepcopy(self.config)

        for i, param in enumerate(self.params):
            next_config['ptycho']['params'][param] = new_x[0,i].item()
        
        return next_config


    def _pr_post_processing(self, X):
        """Apply differentiable rounding to integer dims (PR forward pass)."""
        X_out = X.clone()
        for idx in self.integer_indices:
            # Unnormalize -> approximate_round -> renormalize
            raw = X_out[..., idx] * (self.bounds[1][idx] - self.bounds[0][idx]) + self.bounds[0][idx]
            rounded = approximate_round(raw)
            X_out[..., idx] = (rounded - self.bounds[0][idx]) / (self.bounds[1][idx] - self.bounds[0][idx])
        return X_out


    def tell(self, job_config, y_value):
        config = self.config
        
        x_value = []
        for param in self.params:
            x_value.append(job_config['ptycho']['params'][param])

        self.train_x = np.vstack([
            self.train_x,
            np.array(x_value).reshape(1, -1)
        ])

        self.train_y = np.concatenate([
            self.train_y,
            np.array([y_value])
        ])

        
        train_x_path = config['bo'].get('train_x')
        train_y_path = config['bo'].get('train_y')
        if train_x_path is not None and train_y_path is not None:
            np.save(train_x_path, self.train_x)
            np.save(train_y_path, self.train_y)
        else:
            result_dir = config['io']['result_dir']
            np.save(os.path.join(result_dir, 'train_x.npy'), self.train_x)
            np.save(os.path.join(result_dir, 'train_y.npy'), self.train_y)

    





