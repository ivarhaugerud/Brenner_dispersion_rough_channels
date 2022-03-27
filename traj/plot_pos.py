import numpy as np
import argparse
import os
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
import json
from utils import parse_settings, parse_grid, load_fields
import scipy.optimize as opt


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot/analyze positions beautifully")
    parser.add_argument("folder", type=str, help="Folder")
    parser.add_argument("-Re", type=float, default=0.0, help="Re")
    parser.add_argument("-b", type=float, default=0.3, help="b")
    parser.add_argument("-Dm", type=float, default=0.0, help="Dm")
    parser.add_argument("-U", type=float, default=0.0, help="U")
    parser.add_argument("--skip", type=int, default=1, help="Skip")
    parser.add_argument("--anim", action="store_true", help="Animate")
    args = parser.parse_args()
    return args


def find_subfolder(Re, b, Dm, U, rwfolder):
    for subfolder in os.listdir(rwfolder):
        if "_" in subfolder:
            Re_str, b_str, Dm_str, U_str = subfolder.split("_")
            if bool(Re == float(Re_str[2:]) and
                    b == float(b_str[1:]) and
                    Dm == float(Dm_str[2:]) and
                    U == float(U_str[1:])):
                return subfolder


def func(x, a, b, c):
    return a - b*np.exp(-c*x)


def exp_relax(x, a, b, c):
    return a + b*np.exp(-c*x)


if __name__ == "__main__":
    args = parse_args()

    rwfolder = os.path.join(args.folder, "RandomWalkers")
    subfolder = find_subfolder(args.Re, args.b, args.Dm, args.U, rwfolder)

    if subfolder is None:
        exit("Found no such data.")

    posfolder = os.path.join(rwfolder, subfolder, "Positions")

    Ts = dict([(float(posfile[4:-4]), posfile)
               for posfile in os.listdir(posfolder)])
    t = np.array(sorted(Ts.keys()))
    t = t[::args.skip]

    Nbins = 100
    Pt = np.zeros((Nbins, len(t)))

    snapfolder = os.path.join(posfolder, "Snapshots")
    if args.anim and not os.path.exists(snapfolder):
        os.makedirs(snapfolder)

    sigma2 = np.zeros(len(t))
    xmean = np.zeros(len(t))
    for i, ti in enumerate(t):
        T = Ts[ti]
        pos = np.loadtxt(os.path.join(posfolder, T))

        hist, bins = np.histogram(pos[:, 1]-ti,  # range=[-100, 100],
                                  bins=Nbins, density=False)
        Pt[:, i] = hist

        sigma2[i] = np.var(pos[:, 1])
        xmean[i] = np.mean(pos[:, 1])

        if args.anim:
            fig = plt.figure()
            plt.title("t={}".format(ti))
            plt.scatter(pos[:, 1], pos[:, 2])
            plt.savefig(os.path.join(snapfolder, "{:06d}.png".format(i)))
            plt.close()

    plt.pcolormesh(Pt[:, 2:])
    plt.show()

    it_tr = 1

    t = t[1:]
    sigma2 = sigma2[1:]
    xmean = xmean[1:]-xmean[0]

    popt1 = (1.4, 0.5, 1./500)
    popt1, pcov1 = opt.curve_fit(
        exp_relax, t[1:], xmean[1:]/t[1:], p0=popt1)

    fig1 = plt.figure()
    # plt.plot(t[1:], exp_relax(t[1:], *popt1))
    plt.plot(t[it_tr:], xmean[it_tr:])
    plt.plot(t, args.U*np.ones_like(t))
    plt.xlabel("t")
    plt.ylabel("<x>")
    plt.show()

    U_eff = popt1[0]

    print("t_char =", 1./popt1[2])

    it_start = len(t)//2
    D_eff_app = sigma2[it_start:]/(2*t[it_start])
    D_eff_est1 = D_eff_app.mean()

    popt2 = (D_eff_est1, 1, 1./500)
    popt2, pcov2 = opt.curve_fit(
        func, t[1:], sigma2[1:]/(2*t[1:]), p0=popt2)

    D_eff = popt2[0]

    fig2 = plt.figure()
    plt.plot(t[1:], sigma2[1:]/(2*t[1:]))
    plt.plot(t[1:], func(t[1:], *popt2))
    plt.plot(t[1:], D_eff*np.ones_like(t[1:]))
    plt.xlabel("t")
    plt.ylabel("Var[x]/2t")
    plt.show()

    fig2 = plt.figure()
    plt.plot(t, sigma2)
    plt.plot(t, 2*t*func(t, *popt2))
    plt.plot(t, 2*t*D_eff)
    plt.xlabel("t")
    plt.ylabel("Var[x]")
    plt.show()

    t_char = 1./popt2[2]
    print("t_char =", t_char)
    print("t_max  =", t.max())

    print("U_eff =", U_eff)
    print("D_eff =", D_eff)
    print("std(D_eff) =", )

    Pe = args.U/args.Dm
    print("Pe =", Pe)
    kappa = 2./105
    print("Smooth, theory: D_eff =", args.Dm*(1+kappa*Pe**2))

    print(args.Dm, args.U, D_eff, U_eff, t_char)
