import matplotlib.pyplot as plt
import numpy as np
from meshpy import triangle as tri
import dolfin as df
from dolfin import MPI
import os
import h5py


def mpi_comm():
    return MPI.comm_world


def mpi_barrier():
    MPI.barrier(mpi_comm())


def mpi_rank():
    return MPI.rank(mpi_comm())


def mpi_size():
    return MPI.size(mpi_comm())


def mpi_is_root():
    return mpi_rank() == 0


def remove_safe(path):
    """ Remove file in a safe way. """
    if mpi_is_root() and os.path.exists(path):
        os.remove(path)


def numpy_to_dolfin(nodes, elements, delete_tmp=True):
    """ Convert nodes and elements to a dolfin mesh object. """
    tmpfile = "tmp.h5"

    dim = nodes.shape[1]
    npts = elements.shape[1]
    if dim == 0:
        celltype_str = "point"
    elif dim == 1:
        celltype_str = "interval"
    elif dim == 2:
        if npts == 3:
            celltype_str = "triangle"
        elif npts == 4:
            celltype_str = "quadrilateral"
    elif dim == 3:
        if npts == 4:
            celltype_str = "tetrahedron"
        elif npts == 8:
            celltype_str = "hexahedron"

    if mpi_is_root():
        with h5py.File(tmpfile, "w") as h5f:
            cell_indices = h5f.create_dataset(
                "mesh/cell_indices", data=np.arange(len(elements)),
                dtype='int64')
            topology = h5f.create_dataset(
                "mesh/topology", data=elements, dtype='int64')
            coordinates = h5f.create_dataset(
                "mesh/coordinates", data=nodes, dtype='float64')
            topology.attrs["celltype"] = np.string_(celltype_str)
            topology.attrs["partition"] = np.array([0], dtype='uint64')

    mpi_barrier()

    mesh = df.Mesh()
    h5f = df.HDF5File(mesh.mpi_comm(), tmpfile, "r")
    h5f.read(mesh, "mesh", False)
    h5f.close()

    mpi_barrier()
    if delete_tmp:
        remove_safe(tmpfile)
    return mesh


def dict2list(d):
    l = [[] for _ in range(len(d))]
    for di, i in d.items():
        l[i] = di
    return l

def triangle_rough_mesh(b, dx, flip):
    #flip = bool, 1 or 0, true or false
    if b == 0.0:
        pts_low = [(0.0, 0.0),
                   (0.1, 0.0)]
    else:
        if flip:
            pts_low = [(0.0,   -b/2),
                       (b/2,   -b/2),
                       (b/2,    b/2),
                       (3*b/2, -b/2),
                       (2*b,   -b/2)]

        else: 
            pts_low = [(0.0,   -b/2),
                       (b/2,   -b/2),
                       (3*b/2,  b/2),
                       (3*b/2, -b/2),
                       (2*b,   -b/2)]
    nodes = dict()
    edges = []

    pts = []
    for pt_low in pts_low:
        pts.append((pt_low[0], pt_low[1]-1))

    for pt_low in pts_low[::-1]:
        pts.append((pt_low[0], -pt_low[1]+1))
    pts.append(pts[0])

    for pta, ptb in zip(pts[:-1], pts[1:]):
        ra = np.array(pta)
        rb = np.array(ptb)
        dr = ra-rb
        drabs = np.sqrt(sum(dr**2))
        Nseg = int(np.ceil(drabs/dx))

        beta = np.linspace(0., 1., Nseg)
        rs = [tuple(betai*rb + (1-betai)*ra) for betai in beta]
        for r in rs:
            if r not in nodes:
                nodes[r] = len(nodes)
    
        for ri, rj in zip(rs[:-1], rs[1:]):
            edge = [nodes[ri], nodes[rj]]
            edges.append(edge)

    nodes = dict2list(nodes)

    mi = tri.MeshInfo()
    mi.set_points(nodes)
    mi.set_facets(edges)

    max_area = 0.5*dx**2

    mesh = tri.build(mi, max_volume=max_area, min_angle=25,
                     allow_boundary_steiner=False)

    coords = np.array(mesh.points)
    faces = np.array(mesh.elements)

    msh = numpy_to_dolfin(coords, faces)

    #plot and save for debugging
    #df.plot(msh)
    #plt.savefig("data_square/geometries/triangle_b_"+str(b)+"__flip_"+str(flip)+".png")

    return msh

def saw_tooth_rough_mesh(b, dx, flip):
    #flip = bool, 1 or 0, true or false
    if b == 0.0:
        pts_low = [(0.0, 0.0),
                   (0.1, 0.0)]
    else:
        if flip:
            pts_low = [(0.0,      -b/2),
                       (2*dx,     -b/2),
                       (2*dx,      b/2),
                       (2*b-2*dx, -b/2),
                       (2*b,      -b/2)]

        else: 
            pts_low = [(0.0,      -b/2),
                       (2*dx,     -b/2),
                       (2*b-2*dx,  b/2),
                       (2*b-2*dx, -b/2),
                       (2*b,      -b/2)]
    nodes = dict()
    edges = []

    pts = []
    for pt_low in pts_low:
        pts.append((pt_low[0], pt_low[1]-1))

    for pt_low in pts_low[::-1]:
        pts.append((pt_low[0], -pt_low[1]+1))
    pts.append(pts[0])

    for pta, ptb in zip(pts[:-1], pts[1:]):
        ra = np.array(pta)
        rb = np.array(ptb)
        dr = ra-rb
        drabs = np.sqrt(sum(dr**2))
        Nseg = int(np.ceil(drabs/dx))

        beta = np.linspace(0., 1., Nseg)
        rs = [tuple(betai*rb + (1-betai)*ra) for betai in beta]
        for r in rs:
            if r not in nodes:
                nodes[r] = len(nodes)
    
        for ri, rj in zip(rs[:-1], rs[1:]):
            edge = [nodes[ri], nodes[rj]]
            edges.append(edge)

    nodes = dict2list(nodes)

    mi = tri.MeshInfo()
    mi.set_points(nodes)
    mi.set_facets(edges)

    max_area = 0.5*dx**2

    mesh = tri.build(mi, max_volume=max_area, min_angle=25,
                     allow_boundary_steiner=False)

    coords = np.array(mesh.points)
    faces = np.array(mesh.elements)

    msh = numpy_to_dolfin(coords, faces)

    #plot and saved for debugging
    #df.plot(msh)
    #plt.savefig("data_square/geometries/saw_tooth_b_"+str(b)+"__flip_"+str(flip)+".png")

    return msh

def square_rough_mesh(b, dx):
    if b == 0.0:
        pts_low = [(0.0, 0.0),
                   (0.1, 0.0)]
    else:
        pts_low = [(0.0, -b/2),
                   (b/2, -b/2),
                   (b/2, b/2),
                   (3*b/2, b/2),
                   (3*b/2, -b/2),
                   (2*b, -b/2)]

    nodes = dict()
    edges = []

    pts = []
    for pt_low in pts_low:
        pts.append((pt_low[0], pt_low[1]-1))

    for pt_low in pts_low[::-1]:
        pts.append((pt_low[0], -pt_low[1]+1))
    pts.append(pts[0])

    for pta, ptb in zip(pts[:-1], pts[1:]):
        ra = np.array(pta)
        rb = np.array(ptb)
        dr = ra-rb
        drabs = np.sqrt(sum(dr**2))
        Nseg = int(np.ceil(drabs/dx))

        beta = np.linspace(0., 1., Nseg)
        rs = [tuple(betai*rb + (1-betai)*ra) for betai in beta]
        for r in rs:
            if r not in nodes:
                nodes[r] = len(nodes)
    
        for ri, rj in zip(rs[:-1], rs[1:]):
            edge = [nodes[ri], nodes[rj]]
            edges.append(edge)

    nodes = dict2list(nodes)

    mi = tri.MeshInfo()
    mi.set_points(nodes)
    mi.set_facets(edges)

    max_area = 0.5*dx**2

    mesh = tri.build(mi, max_volume=max_area, min_angle=25,
                     allow_boundary_steiner=False)

    coords = np.array(mesh.points)
    faces = np.array(mesh.elements)

    msh = numpy_to_dolfin(coords, faces)

    return msh

def fractal_1_square_rough_mesh(b, dx):
    if b == 0.0:
        pts_low = [(0.0, 0.0),
                   (0.1, 0.0)]
    else:
        pts_low = [(0.0,    -b/6),
                   (b/6,    -b/6),
                   (b/6,    -b/2),
                   (b/2,    -b/2),
                   (b/2,     b/2),
                   (5*b/6,   b/2),
                   (5*b/6,   b/6),
                   (7*b/6,   b/6),
                   (7*b/6,   b/2),
                   (3*b/2,   b/2),
                   (3*b/2,  -b/2),
                   (11*b/6, -b/2),
                   (11*b/6, -b/6),
                   (2*b,    -b/6)]

    nodes = dict()
    edges = []

    pts = []
    for pt_low in pts_low:
        pts.append((pt_low[0], pt_low[1]-1))

    for pt_low in pts_low[::-1]:
        pts.append((pt_low[0], -pt_low[1]+1))
    pts.append(pts[0])

    for pta, ptb in zip(pts[:-1], pts[1:]):
        ra = np.array(pta)
        rb = np.array(ptb)
        dr = ra-rb
        drabs = np.sqrt(sum(dr**2))
        Nseg = int(np.ceil(drabs/dx))

        beta = np.linspace(0., 1., Nseg)
        rs = [tuple(betai*rb + (1-betai)*ra) for betai in beta]
        for r in rs:
            if r not in nodes:
                nodes[r] = len(nodes)
    
        for ri, rj in zip(rs[:-1], rs[1:]):
            edge = [nodes[ri], nodes[rj]]
            edges.append(edge)

    nodes = dict2list(nodes)

    mi = tri.MeshInfo()
    mi.set_points(nodes)
    mi.set_facets(edges)

    max_area = 0.5*dx**2

    mesh = tri.build(mi, max_volume=max_area, min_angle=25,
                     allow_boundary_steiner=False)

    coords = np.array(mesh.points)
    faces = np.array(mesh.elements)

    msh = numpy_to_dolfin(coords, faces)

    return msh

def fractal_2_square_rough_mesh(b, dx):
    if b == 0.0:
        pts_low = [(0.0, 0.0),
                   (0.1, 0.0)]
    else:
        pts_low = [(0.0,     -5*b/18),
                   (b/18,    -5*b/18),
                   (b/18,    -b/6),
                   (b/6,     -b/6),
                   (b/6,     -b/2),
                   (5*b/18,  -b/2),
                   (5*b/18, -7*b/18),
                   (7*b/18, -7*b/18),
                   (7*b/18,  -b/2),
                   (b/2,     -b/2),
                   (b/2,     b/2),
                   (11*b/18, b/2),
                   (11*b/18, 7*b/18),
                   (13*b/18, 7*b/18),
                   (13*b/18, b/2),
                   (5*b/6,   b/2),
                   (5*b/6,   b/6),
                   (17*b/18, b/6),
                   (17*b/18, 5*b/18),
                   (19*b/18, 5*b/18),
                   (19*b/18, b/6),
                   (7*b/6,   b/6),
                   (7*b/6,   b/2),
                   (23*b/18, b/2),
                   (23*b/18, 7*b/18),
                   (25*b/18, 7*b/18),
                   (25*b/18, b/2),
                   (3*b/2,   b/2),
                   (3*b/2,  -b/2),
                   (29*b/18, -b/2),
                   (29*b/18, -7*b/18),
                   (31*b/18, -7*b/18),
                   (31*b/18, -b/2),
                   (11*b/6, -b/2),
                   (11*b/6, -b/6),
                   (35*b/18, -b/6),
                   (35*b/18, -5*b/18),
                   (2*b,    -5*b/18)]

    nodes = dict()
    edges = []

    pts = []
    for pt_low in pts_low:
        pts.append((pt_low[0], pt_low[1]-1))

    for pt_low in pts_low[::-1]:
        pts.append((pt_low[0], -pt_low[1]+1))
    pts.append(pts[0])

    for pta, ptb in zip(pts[:-1], pts[1:]):
        ra = np.array(pta)
        rb = np.array(ptb)
        dr = ra-rb
        drabs = np.sqrt(sum(dr**2))
        Nseg = int(np.ceil(drabs/dx))

        beta = np.linspace(0., 1., Nseg)
        rs = [tuple(betai*rb + (1-betai)*ra) for betai in beta]
        for r in rs:
            if r not in nodes:
                nodes[r] = len(nodes)
    
        for ri, rj in zip(rs[:-1], rs[1:]):
            edge = [nodes[ri], nodes[rj]]
            edges.append(edge)

    nodes = dict2list(nodes)

    mi = tri.MeshInfo()
    mi.set_points(nodes)
    mi.set_facets(edges)

    max_area = 0.5*dx**2

    mesh = tri.build(mi, max_volume=max_area, min_angle=25,
                     allow_boundary_steiner=False)

    coords = np.array(mesh.points)
    faces = np.array(mesh.elements)

    msh = numpy_to_dolfin(coords, faces)

    return msh

if __name__ == "__main__":
    dx = 0.02
    b = 0.5
    
    msh = saw_tooth_rough_mesh(b, dx, False)
