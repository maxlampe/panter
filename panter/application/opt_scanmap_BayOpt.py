""""""

import numpy as np
from panter.core.scan2DMap import ScanMapClass
from panter.config.filesScanMaps import scan_200117

import torch
import torch.autograd as autograd
import torch.optim as optim
from torch.distributions import constraints, transform_to

import pyro
import pyro.contrib.gp as gp
import warnings

warnings.filterwarnings("ignore")
assert pyro.__version__.startswith("1.7.0")


def const_weights(weights: np.array):
    w_list = [1.0] * 16

    if weights.shape[0] == 4:
        for i in range(4):
            w_list[(i*2)] = weights[i]
            w_list[(i*2 + 1)] = weights[i]
    elif weights.shape[0] == 8:
        for i in range(4):
            w_list[i] = weights[i]
    else:
        assert False, "ERROR: Invalid weight array length. Needs to be 4 or 8."

    return np.array(w_list)


def main(
    n_start_data: int = 20,
    n_opt_steps: int = 10,
    n_candidates: int = 10,
    w_range: np.array = np.array([0.8, 1.2]),
    b_dummy_val: bool = False,
):
    pos, evs = scan_200117()
    smc = ScanMapClass(
        scan_pos_arr=pos,
        event_arr=evs,
        detector=0,
    )

    pyro.clear_param_store()
    # Initialize with random data points
    x = []
    y = []
    for i in range(n_start_data):
        # rng_weights = np.random.rand(4) * (w_range[1] - w_range[0]) + w_range[0]
        rng_weights = 0.08 * np.random.randn(4) + 1.0
        weights = const_weights(rng_weights)

        if b_dummy_val:
            losses = [0.0, np.random.rand(1) * 30000 + 5000]
        else:
            smc.calc_peak_positions(weights=weights)
            losses = smc.calc_loss()

        if losses[1] is not None:
            x.append(rng_weights)
            y.append(losses[1])

    x = torch.tensor(x)
    y = torch.flatten(torch.tensor(y))

    print(x)
    print(y)

    gpmodel = gp.models.GPRegression(
        x, y, gp.kernels.Matern52(input_dim=4), noise=torch.tensor(0.1), jitter=1.0e-4
    )

    def update_posterior(x_new):
        if b_dummy_val:
            losses = [0.0, np.random.rand(1) * 30000 + 5000]
        else:
            new_weights = const_weights(torch.flatten(x_new))
            smc.calc_peak_positions(weights=np.array(new_weights))
            losses = smc.calc_loss()

        if losses[1] is not None:
            y = torch.tensor([losses[1]])
            print("Curr losses ", losses)

            x = torch.cat([gpmodel.X, x_new])  # incorporate new evaluation
            y = torch.cat([gpmodel.y, y])

            print("updated x ", x)
            print("updated y ", y)

            gpmodel.set_data(x, y)
            # optimize the GP hyperparameters using Adam with lr=0.001
            optimizer = torch.optim.Adam(gpmodel.parameters(), lr=0.001)
            gp.util.train(gpmodel, optimizer)

    def lower_confidence_bound(x, kappa=3.0):
        mu, variance = gpmodel(x, full_cov=False, noiseless=False)
        sigma = variance.sqrt()
        return mu - kappa * sigma

    def find_a_candidate(x_init, lower_bound=w_range[0], upper_bound=w_range[1]):
        # transform x to an unconstrained domain
        constraint = constraints.interval(lower_bound, upper_bound)
        unconstrained_x_init = transform_to(constraint).inv(x_init)
        unconstrained_x = unconstrained_x_init.clone().detach().requires_grad_(True)
        minimizer = optim.LBFGS([unconstrained_x], line_search_fn="strong_wolfe")

        def closure():
            minimizer.zero_grad()
            x = transform_to(constraint)(unconstrained_x)
            y = lower_confidence_bound(x)
            autograd.backward(unconstrained_x, autograd.grad(y, unconstrained_x))
            return y

        minimizer.step(closure)
        # after finding a candidate in the unconstrained domain,
        # convert it back to original domain.
        x2 = transform_to(constraint)(unconstrained_x)
        return x2.detach()

    def next_x(
        lower_bound=w_range[0], upper_bound=w_range[1], num_candidates=n_candidates
    ):
        candidates = []
        values = []


        # Start with best candidate x and sample rest random
        argmin = torch.min(gpmodel.y, dim=0)[1].item()
        x_init = torch.unsqueeze(gpmodel.X[argmin], 0)
        for i in range(num_candidates):
            x = find_a_candidate(x_init, lower_bound, upper_bound)
            y = lower_confidence_bound(x)
            candidates.append(x)
            values.append(y)
            # DIM!
            x_init = x.new_empty((1, 4)).uniform_(lower_bound, upper_bound)

        # Use minimum (best) result
        print("candidates ", candidates)
        print("values", values)
        argmin = torch.min(torch.cat(values), dim=0)[1].item()
        print("winner ", candidates[argmin])
        return candidates[argmin]

    optimizer = torch.optim.Adam(gpmodel.parameters(), lr=0.001)
    gp.util.train(gpmodel, optimizer)
    for i in range(n_opt_steps):
        xmin = next_x()
        update_posterior(xmin)


if __name__ == "__main__":
    main(
        b_dummy_val=False,
        n_start_data=20,
        n_opt_steps=20,
        n_candidates=20,
    )

# [0.9966, 0.9748, 0.9987, 0.9900] 8593.5421
# [1.0080, 0.9590, 0.9894, 1.0032] 8188.6815