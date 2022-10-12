""""""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import os
import matplotlib.dates as md

import torch
import pyro
import pyro.contrib.gp as gp
import pyro.distributions as dist

from panter.config import conf_path
from panter.map.drift_dettsum_map import DriftDetSumMapPerkeo

assert pyro.__version__.startswith("1.7.0")
pyro.set_rng_seed(0)
plt.rcParams.update({"font.size": 12})
output_path = os.getcwd()


class GPRDriftData:
    """"""

    def __init__(self, n_data_steps: int = None):
        self._n_data_steps = n_data_steps
        self.pdm = DriftDetSumMapPerkeo(np.array([]), bimp_detsum=True, bimp_sn=False)
        self.pdm()

        self.df = self.pdm.maps[0]
        self.data_form = {"a": None, "b": None}
        self.x, self.y0, self.y1 = self._preprocess_data()

    def __call__(self, *args, **kwargs):
        # return {"time": self.x[1:], "det0": self.y0[1:], "det1": self.y1[1:]}
        return {"time": self.x, "det0": self.y0, "det1": self.y1}

    def _preprocess_data(self):
        """Inplace modification"""

        for index, row in self.df.iterrows():
            if pd.isnull(row["pmt_fac"][1]):
                self.df.drop([index], inplace=True)

        # Normalize data
        x_raw = np.array(self.df["time"].apply(pd.Series))
        self.data_form["a"] = x_raw[0]
        self.data_form["b"] = x_raw[-1] - x_raw[0]

        x_raw = self.timestamp_to_data(x_raw)
        y_raw0 = np.array(self.df["pmt_fac"].apply(pd.Series)[0])
        y_raw1 = np.array(self.df["pmt_fac"].apply(pd.Series)[1])

        x = torch.flatten(torch.tensor(x_raw[:: self._n_data_steps]))
        y0 = torch.tensor(y_raw0[:: self._n_data_steps])
        y1 = torch.tensor(y_raw1[:: self._n_data_steps])

        return x, y0, y1

    def data_to_timestamp(self, x: torch.tensor):
        """"""
        return x * self.data_form["b"] + self.data_form["a"]

    def timestamp_to_data(self, x: torch.tensor):
        """"""
        return (x - self.data_form["a"]) / self.data_form["b"]


class GPRDrift:
    def __init__(self, detector: int = 0):
        self._det = detector
        self.dataclass = GPRDriftData()
        self.data = self.dataclass()
        self.gpr = self._create_gpr_model(
            x_train=self.data["time"], y_train=self.data[f"det{self._det}"]
        )
        # ^ minimum for enable loading of state dict

        self.losses = []

    def __call__(self, x_eval: torch.tensor):
        """"""

        self.gpr.eval()
        with torch.no_grad():
            mean, cov = self.gpr(x_eval.double(), full_cov=True, noiseless=False)
        sd = cov.diag().sqrt()

        return mean, sd

    def train(self, n_steps: int = 7000):
        """"""

        self.gpr.train()

        optimizer = torch.optim.Adam(self.gpr.parameters(), lr=0.005)
        loss_fn = pyro.infer.Trace_ELBO().differentiable_loss
        for i in range(n_steps):
            optimizer.zero_grad()
            loss = loss_fn(self.gpr.model, self.gpr.guide)
            loss.backward()
            optimizer.step()
            self.losses.append(loss.item())

    def plot_results(
        self, n_test: int = 500, t_range=None, y_lim=None, bsave=False, file_tag=None
    ):
        """"""

        fig, ax = plt.subplots(figsize=(10, 6))
        x_data = self.data["time"]
        y_data = self.data[f"det{self._det}"]
        x_test = torch.linspace(
            x_data.numpy().min() - 0.03,
            x_data.numpy().max() + 0.03,
            n_test,
            dtype=torch.float64,
        )
        x_plot = self.dataclass.data_to_timestamp(x_test.numpy())
        mean, sd = self(x_test)

        xfmt = md.DateFormatter("%m-%d\n%H:%M")
        ax.xaxis.set_major_formatter(xfmt)
        dates_plot = [datetime.datetime.fromtimestamp(t) for t in x_plot]
        x_data_plot = [
            datetime.datetime.fromtimestamp(t)
            for t in self.dataclass.data_to_timestamp(x_data.numpy())
        ]

        # ax.plot(x_data_plot, y_data.numpy(), "kx", label="Drift data")
        # ax.plot(dates_plot, mean.numpy(), "r", lw=2, label="GPR")
        ax.plot(x_data_plot, y_data.numpy(), "x", c="#0C0887", label="Drift data")
        ax.plot(dates_plot, mean.numpy(), c="#FF220C", lw=2, label="GPR")
        ax.fill_between(
            dates_plot,  # plot the two-sigma uncertainty about the mean
            (mean - 2.0 * sd).numpy(),
            (mean + 2.0 * sd).numpy(),
            # color="C0",
            color="#FDC328",
            alpha=0.3,
            label="GPR +- 2\u03C3",
        )
        if t_range is not None:
            t_range = self.dataclass.data_to_timestamp(t_range)
            t_range = [datetime.datetime.fromtimestamp(t) for t in t_range]
            # ax.set_xlim(t_range)
        if y_lim is not None:
            ax.set_ylim(y_lim)

        # ax.set_title("Drift correction", fontsize=18)
        # ax.set(xlabel="Time [M - D / T]", ylabel="Drift factor [a.u.]")
        ax.set(xlabel="Time t [M - D / T]", ylabel="Correction c_T [a.u.]")
        ax.legend()
        plt.tight_layout()
        if bsave:
            if file_tag is None:
                file_tag = ""
            else:
                file_tag = "_" + file_tag
            plt.savefig(output_path + "/" + f"drift_gpr{file_tag}.png", dpi=300)
        plt.show()

    def plot_losses(self):
        """"""
        plt.plot(self.losses)
        plt.show()

    def save_model(self, file_name: str = "gpr_model.plk"):
        """"""
        torch.save(self.gpr.state_dict(), file_name)

    def load_model(self, file_name: str = "gpr_model.plk"):
        """"""
        self.gpr.load_state_dict(torch.load(file_name))

    @staticmethod
    def _create_gpr_model(x_train: torch.tensor, y_train: torch.tensor):
        """"""

        pyro.clear_param_store()
        kernel = gp.kernels.Matern32(
            input_dim=1,
            variance=torch.tensor(1.0),
            lengthscale=torch.tensor(0.01),
        )
        gpr = gp.models.GPRegression(
            X=x_train,
            y=y_train,
            kernel=kernel,
            noise=torch.tensor(1.0),
            mean_function=lambda x: torch.tensor(1.0),
        )

        gpr.kernel.lengthscale = pyro.nn.PyroSample(dist.LogNormal(0.0, 1.0))
        gpr.kernel.variance = pyro.nn.PyroSample(dist.LogNormal(0.0, 1.0))

        return gpr


def main(bcalc_anew: bool = False):

    for det in [0, 1]:
        gpr_class = GPRDrift(detector=det)

        if bcalc_anew:
            gpr_class.train()
            gpr_class.save_model(file_name=f"{conf_path}/gpr_model_det{det}.plk")

        else:
            gpr_class.load_model(file_name=f"{conf_path}/gpr_model_det{det}.plk")

        gpr_class.plot_results(t_range=[0.0, 1.0], bsave=True, file_tag=f"det{det}")
        res = gpr_class(torch.tensor([0.1, 0.3, 0.8]))
        print(res)


if __name__ == "__main__":
    main(bcalc_anew=True)
