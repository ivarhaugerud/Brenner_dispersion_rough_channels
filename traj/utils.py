import numpy as np
import json


def parseval(value):
    try:
        value = json.loads(value)
    except ValueError:
        # json understands true/false, not True/False
        if value in ["True", "False"]:
            value = eval(value)
        elif "True" in value or "False" in value:
            value = eval(value)
    return value


def parse_settings(infile):
    fp = open(infile, "r")
    kwargs = dict()
    s = fp.readline()
    count = 1
    while s:
        if s.count('=') == 1:
            key, value = s.strip().split('=', 1)
        else:
            raise TypeError(
                "Only kwargs separated with at the most a single '=' allowed.")

        value = parseval(value)
        kwargs[key] = value

        s = fp.readline()
        count += 1

    return kwargs


def parse_grid(infile):
    fp = open(infile, "r")
    s = fp.readline()
    count = 1

    grid = []
    while s:
        grid.append([si == "f" for si in s.strip().split(" ")])
        s = fp.readline()
        count += 1

    return np.array(grid, dtype=bool)


def load_fields(infile, rho, u_x, u_y):
    fp = open(infile, "r")
    s = fp.readline()
    count = 1
    while s:
        ix, iy, rhoi, mi_x, mi_y = s.strip().split(" ")
        s = fp.readline()
        count += 1

        ix = int(ix)
        iy = int(iy)
        rhoi = float(rhoi)
        mi_x = float(mi_x)
        mi_y = float(mi_y)

        rho[iy, ix] = rhoi
        u_x[iy, ix] = mi_x/rhoi
        u_y[iy, ix] = mi_y/rhoi
