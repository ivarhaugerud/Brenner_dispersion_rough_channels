import numpy as np
import argparse
import os
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
import json
from utils import parse_settings, parse_grid, load_fields


def parse_args():
    parser = argparse.ArgumentParser(description="Plot tracer particles beatifully")
    parser.add_argument("folder", type=str, help="Folder")
    parser.add_argument("-Pe", type=float, default=0, help="Peclet number")
    args = parser.parse_args()
    return args


def find_subfolder(Pe, rwfolder):
    for subfolder in os.listdir(rwfolder):
        if Pe == float(subfolder[2:]):
            return subfolder


if __name__ == "__main__":
    args = parse_args()

    settingsfile = os.path.join(args.folder, "Settings/config.case")
    settings = parse_settings(settingsfile)

    tsfolder = os.path.join(args.folder, "Timeseries")
    rwfolder = os.path.join(args.folder, "RandomWalkers")

    grid = parse_grid(settings["grid"])
    not_grid = np.logical_not(grid)

    ny, nx = grid.shape

    its = []
    for filename in os.listdir(tsfolder):
        if filename[:7] == "fields_" and filename[-4:] == ".dat":
            its.append(int(filename[7:-4]))
    its = sorted(its)
    it = its[-1]
    filename = os.path.join(tsfolder, "fields_{}.dat".format(it))

    rho = np.zeros((ny, nx))
    u_x = np.zeros((ny, nx))
    u_y = np.zeros((ny, nx))

    load_fields(filename, rho, u_x, u_y)

    subfolder = find_subfolder(args.Pe, rwfolder)

    trajfolder = os.path.join(rwfolder, subfolder, "Trajectories")

    plt.figure()
    for trajfile in os.listdir(trajfolder):
        traj = np.loadtxt(os.path.join(trajfolder, trajfile))
        plt.plot(traj[:, 1], traj[:, 2])

    plt.show()
