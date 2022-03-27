import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker, cm


parser = argparse.ArgumentParser(description="Analyze")
parser.add_argument("folder", type=str, help="Folder")
args = parser.parse_args()

files = os.listdir(args.folder)
data = dict()
for f in files:
    if f[-4:] == ".dat" and "Re" in f and "b" in f:
        for kw in f[:-4].split("_"):
            if kw[:2] == "Re":
                Re = float(kw[2:])
            if kw[:1] == "b":
                b = float(kw[1:])
        print(Re, b)
        if Re not in data:
            data[Re] = dict()
        data[Re][b] = np.loadtxt(os.path.join(args.folder, f))

Re = 0.0
# b, Pe
b_bPe = np.zeros((len(data[Re]), len(data[Re][b])))
Pe_bPe = np.zeros((len(data[Re]), len(data[Re][b])))
D_bPe = np.zeros((len(data[Re]), len(data[Re][b])))
for i, key in enumerate(sorted(data[Re].keys())):
    size = len(data[Re][key])
    print(key, size)
    b_bPe[i, :] = key
    Pe_bPe[i, :size] = data[Re][key][:, 0]
    D_bPe[i, :size] = data[Re][key][:, 1]

fig, ax = plt.subplots()
cc = ax.contourf(np.log10(Pe_bPe), b_bPe, np.log10(D_bPe),
                 locator=ticker.LinearLocator(numticks=30,))
cbar = fig.colorbar(cc)
plt.xlabel("log10(Pe)")
plt.ylabel("b")


g = (D_bPe-1)/Pe_bPe**2/(2.0/105)
g_p = np.ma.masked_where(g < 0, g)
g_m = np.ma.masked_where(g >= 0, g)

fig, ax = plt.subplots()
cs1 = ax.contourf(np.log10(Pe_bPe), b_bPe, g_p, cmap="Reds",
                  locator=ticker.LinearLocator(numticks=30,))
cs2 = ax.contourf(np.log10(Pe_bPe), b_bPe, np.log10(-g_m), cmap="Blues",
                  locator=ticker.LinearLocator(numticks=30,))
plt.colorbar(cs1)
plt.colorbar(cs2)


#c1 = plt.pcolormesh(np.log10(Pe_bPe), b_bPe, g_p, cmap="Reds")
#c2 = plt.pcolormesh(np.log10(Pe_bPe), b_bPe, -g_m, cmap="Blues")
#cbar = plt.colorbar(c1)
#plt.colorbar(c2)
#cbar.solids.set_clim(0, 50)

plt.show()
