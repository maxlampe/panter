"""Plotting ot ElecTest results."""

import numpy as np
import matplotlib.pyplot as plt

# import matplotlib as mpl
# from mpl_toolkits.mplot3d import Axes3D
from lmfit import Model

import evalPerkeo as eP


test5vals1 = [
    0.5 * (22.439324883868494 + 23.26855886150173),
    0.5 * (9.149675298096554 + 14.021974161966746),
]
test5vals2 = [
    0.5 * (0.0058787181539636645 + 0.0056685667499115773),
    0.5 * (0.0009473376280875119 + 0.0009166678753262123),
]

t5_rel_200ue500 = 0.5 * np.array(
    [
        0.0022505028319746946 + 0.0022505028319746946,
        0.0002710323672066126 + 0.00026701638374428,
    ]
)
t5_rel_500ue200 = 0.5 * np.array(
    [
        0.00240153174219454 + 0.00230850867799267,
        0.000846839760925564 + 0.000251107338791139,
    ]
)

t5_rel_050ue300 = 0.5 * np.array(
    [
        0.002254524436417994 + 0.002254524436417994,
        0.00022931069142007886 + 0.00022931069142007886,
    ]
)
t5_rel_300ue050 = 0.5 * np.array(
    [
        0.002340241612109116 + 0.0022510408934452953,
        0.0008468397609255636 + 0.0008632157426719546,
    ]
)

print(t5_rel_200ue500)
print(t5_rel_500ue200)
print(t5_rel_050ue300)
print(t5_rel_300ue050)


if True:
    fig = plt.figure()
    ax = fig.gca(projection="3d")

    # Scatterplot
    if False:
        pts = [
            (50, 300, test5vals1[1]),
            (200, 200, 9.156232705166758),
            (200, 500, test5vals1[0]),
            (300, 50, test5vals1[1]),
            (300, 300, 14.021761019594578),
            (500, 200, test5vals1[0]),
            (500, 500, 23.315060763750694),
        ]
        for p in pts:
            ax.scatter(p[0], p[1], p[2], zdir="z", c="r")

    # Triangulated Surfplot
    if True:
        pts2 = np.array(
            [
                (50, 300, test5vals1[1]),
                (200, 200, 9.156232705167),
                (200, 500, test5vals1[0]),
                (300, 50, test5vals1[1]),
                (300, 300, 14.021761019594578),
                (500, 200, test5vals1[0]),
                (500, 500, 23.315060763750694),
            ]
        )
        ax.plot_trisurf(pts2.T[0], pts2.T[1], pts2.T[2], alpha=0.9)
        ax.set_zlabel("abs Dev [mV]")
    else:
        pts3 = np.array(
            [
                (50, 300, t5_rel_050ue300[0]),
                (200, 200, 0.0022999386892712634),
                (200, 500, t5_rel_200ue500[0]),
                (300, 50, t5_rel_300ue050[0]),
                (300, 300, 0.0023529427010620077),
                (500, 200, t5_rel_500ue200[0]),
                (500, 500, 0.002345863343340991),
            ]
        )
        ax.plot_trisurf(pts3.T[0], pts3.T[1], pts3.T[2], alpha=0.9)
        ax.set_zlabel("rel Dev [ ]")

    ax.set_xlabel("Ampl 1 [mV]")
    ax.set_ylabel("Ampl 2 [mV]")
    plt.title("Electronics amplitude dependency")
    plt.show()


ampl_t4 = [200, 300, 500]

a4 = np.array([9.156232705166758, 14.021761019594578, 23.315060763750694])
aerr4 = np.array([0.7586474541332234, 1.3560299400722782, 2.195893422958855])
arrAbsDiff_4 = [a4, aerr4]
r4 = np.array([0.0022999386893, 0.0023529427, 0.0023459])
rerr4 = np.array([0.00021938, 0.0002537977, 0.00021047])
arrRelDiff_4 = [r4, rerr4]
print(arrRelDiff_4)


ampl_t4_long = [50, 205, 200, 305, 300, 505, 500]
r4_long = np.array(
    [
        t5_rel_050ue300[0],
        0.0022999386893,
        t5_rel_200ue500[0],
        0.0023529427,
        t5_rel_300ue050[0],
        0.0023459,
        t5_rel_500ue200[0],
    ]
)
rerr4_long = np.array(
    [
        t5_rel_050ue300[1],
        0.00021938,
        t5_rel_200ue500[1],
        0.0002537977,
        t5_rel_300ue050[1],
        0.00021047,
        t5_rel_500ue200[1],
    ]
)
arrRelDiff_4_long = [r4_long, rerr4_long]


def lin(x, a, b):
    """Simple, linear function in x."""
    return a * np.array(x, dtype=float) + b


def plot_ratedep(
    indat={
        "ampl": ampl_t4,
        "ampl_long": ampl_t4_long,
        "abs": a4,
        "abserr": aerr4,
        "rel": r4_long,
        "relerr": rerr4_long,
    },
    plt_title=("Amplitude based rate dependency" "result (same Ampl.)"),
):
    """Plot rate dependence results"""

    gmodel = Model(lin)
    params = gmodel.make_params(a=1.0, b=0.0)

    fitres_abs = gmodel.fit(
        indat["abs"], params, x=indat["ampl"], weights=eP.calc_weights(indat["abserr"])
    )
    params["a"].vary = False
    params["a"].value = 0.0
    fitres_rel = gmodel.fit(
        indat["rel"],
        params,
        x=indat["ampl_long"],
        weights=eP.calc_weights(indat["relerr"]),
    )
    # print(fitres_abs.fit_report())
    # print(fitres_rel.fit_report())

    fig, axs = plt.subplots(1, 2, figsize=(15, 8))
    fig.suptitle(plt_title)
    axs[0].set_title("Abs Deviation")
    axs[0].errorbar(indat["ampl"], indat["abs"], yerr=indat["abserr"], fmt=".")
    axs[0].set_ylim([0, 30])
    axs[0].plot(
        indat["ampl"],
        gmodel.func(indat["ampl"], **fitres_abs.params),
        "r-",
        label="best fit",
    )
    axs[0].annotate(
        f" f(x) = a*x + b \n"
        f'    a = {fitres_abs.params["a"].value:.4f} +- '
        f'{fitres_abs.params["a"].stderr:.4f}\n'
        f'    b = {fitres_abs.params["b"].value:.4f} +- '
        f'{fitres_abs.params["b"].stderr:.4f}',
        xy=(0.05, 0.95),
        xycoords="axes fraction",
        ha="left",
        va="top",
        bbox=dict(boxstyle="round", fc="1"),
    )
    axs[1].set_title("Rel Deviation")
    axs[1].errorbar(
        indat["ampl_long"], indat["rel"] * 100.0, yerr=indat["relerr"] * 100.0, fmt="."
    )
    axs[1].set_ylim([0, 0.4])
    axs[1].plot(
        indat["ampl_long"],
        100.0 * gmodel.func(indat["ampl_long"], **fitres_rel.params),
        "r-",
        label="best fit",
    )
    axs[1].annotate(
        f" f(x) = b \n"
        f'    b = {100*fitres_rel.params["b"].value:.6f} +- '
        f'{100*fitres_rel.params["b"].stderr:.6f} [%]',
        xy=(0.05, 0.95),
        xycoords="axes fraction",
        ha="left",
        va="top",
        bbox=dict(boxstyle="round", fc="1"),
    )
    axs.flat[0].set(xlabel="Ampltiude [mV]", ylabel="Abs deviation [ch]")
    axs.flat[1].set(xlabel="Ampltiude [mV]", ylabel="Rel deviation [%]")
    plt.show()

    return 0


plot_ratedep()


"""

Test4 @34mus
200
Abs Diff:  9.156232705166758 	 0.7586474541332234
Rel Diff:  0.0022999386892712634 	 0.00021937900965383618


300
Abs Diff:  14.021761019594578 	 1.3560299400722782
Rel Diff:  0.0023529427010620077 	 0.00025379769046411155


500
Abs Diff:  23.315060763750694 	 2.195893422958855
Rel Diff:  0.002345863343340991 	 0.00021047004095909978


Test5 @32

200500

Abs Diff:  23.26855886150173 	 2.4791594089573015
Rel Diff:  0.0058787181539636645 	 0.0007443719755880919
Rel Diff to other:  0.0022505028319746946 	 0.00026701638374428

Abs Diff:  9.454358294696135 	 0.9501222536211971
Rel Diff:  0.0009473376280875119 	 0.00010485427912646562
Rel Diff to other:  0.00240153174219454 	 0.0002710323672066126

050300

Abs Diff:  14.021974161966746 	 1.3206141337782382
Rel Diff:  0.013969677326635049 	 0.0016833518380247102
Rel Diff to other:  0.002254524436417994 	 0.00022931069142007886

Abs Diff:  2.347021428412461 	 0.8561542149751449
Rel Diff:  0.00038871397141542006 	 0.00012999243102638375
Rel Diff to other:  0.002340241612109116 	 0.0008468397609255636

Test5 @36

200500

Abs Diff:  22.439324883868494 	 2.271607079654169
Rel Diff:  0.005668566749911577 	 0.0006868679568683885
Rel Diff to other:  0.0022505028319746946 	 0.00026701638374428

Abs Diff:  9.149675298096554 	 0.9015858322358292
Rel Diff:  0.0009166678753262123 	 9.836989708550025e-05
Rel Diff to other:  0.00230850867799267 	 0.0002511073387911394


050300

Abs Diff:  13.526098097605612 	 1.1819484153913182
Rel Diff:  0.013475082338792603 	 0.0015547188376210093
Rel Diff to other:  0.002254524436417994 	 0.00022931069142007886


Abs Diff:  2.2809831042912947 	 0.9666612937899882
Rel Diff:  0.0003774025514289565 	 0.00014605236541117124
Rel Diff to other:  0.0022510408934452953 	 0.0008632157426719546


"""
