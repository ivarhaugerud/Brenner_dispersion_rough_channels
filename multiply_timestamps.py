import numpy as np
import argparse
import os
import matplotlib.pyplot as plt
from shutil import copyfile, move
from scipy.interpolate import CubicSpline
from scipy.integrate import quad


parser = argparse.ArgumentParser(description="Hey")
parser.add_argument("basefolder", type=str, help="Basefolder")
parser.add_argument("--t_max", type=float, default=1e6, help="Maximum time T")
args = parser.parse_args()

tsfilename = os.path.join(args.basefolder, "timestamps.dat")
tsfilename_new = os.path.join(args.basefolder, "timestamps_new.dat")
tsfilename_short = os.path.join(args.basefolder, "timestamps_short.dat")

#copyfile(tsfilename, tsfilename_short)
with open(tsfilename, "r") as ifile:
    lines = ifile.read().split("\n")
    t = []
    filenames = []
    for line in lines:
        if " " in line:
            t_loc, filename_loc = line.split(" ")
            t.append(float(t_loc))
            filenames.append(filename_loc)

dt = t[1]-t[0]
T = t[-1] + dt
with open(tsfilename_new, "w") as ofile:
    t_loc = t[0]
    while t_loc <= args.t_max:
        for ti, fi in zip(t, filenames):
            ofile.write("{} {}\n".format(t_loc+ti, fi))
        t_loc = t_loc + T

move(tsfilename, tsfilename_short)
move(tsfilename_new, tsfilename)
