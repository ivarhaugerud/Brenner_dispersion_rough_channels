import dolfin as df
from mesh_box import square_rough_mesh
from bcs import PeriodicBC, Wall, NotWall
import numpy as np
import argparse
import os
from mpi4py import MPI

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
folder = "data_square"


# Form compiler options
df.parameters["form_compiler"]["optimize"] = True
df.parameters["form_compiler"]["cpp_optimize"] = True

parser = argparse.ArgumentParser(description="Pure diffusion")
parser.add_argument("-res", type=int, default=32, help="Resolution")
parser.add_argument("-b_min", type=float, default=0.0,
                    help="Roughness b min")
parser.add_argument("-b_max", type=float, default=1.5,
                    help="Roughness b max")
parser.add_argument("-b_N", type=int, default=15,
                    help="Roughness b N")
args = parser.parse_args()

N = args.res
Ly = 2.0
dx = Ly/N

data = np.zeros((args.b_N, 2))
for ib, b in enumerate(np.linspace(
        args.b_min, args.b_max, args.b_N)):
    # Change this
    mesh = square_rough_mesh(b, dx)
    coords = mesh.coordinates()[:]
    Lx = df.MPI.max(df.MPI.comm_world, coords[:, 0].max())

    Echi = df.FiniteElement("Lagrange", mesh.ufl_cell(), 2)

    pbc = PeriodicBC(Lx, Ly)
    wall = Wall(Lx, Ly)
    notwall = NotWall(Lx, Ly)

    subd = df.MeshFunction("size_t", mesh, mesh.topology().dim()-1)
    subd.set_all(0)
    wall.mark(subd, 1)
    notwall.mark(subd, 0)

    S = df.FunctionSpace(mesh, Echi, constrained_domain=pbc)

    one = df.interpolate(df.Constant(1.), S)
    V_Omega = df.assemble(one*df.dx)

    n = df.FacetNormal(mesh)

    chi = df.TrialFunction(S)
    chi_ = df.Function(S, name="chi")
    psi = df.TestFunction(S)

    ds = df.Measure("ds", domain=mesh, subdomain_data=subd)

    F_chi = (n[0]*psi*ds(1)
             + df.inner(df.grad(chi), df.grad(psi))*df.dx)

    a_chi, L_chi = df.lhs(F_chi), df.rhs(F_chi)

    problem_chi2 = df.LinearVariationalProblem(a_chi, L_chi, chi_, bcs=[])
    solver_chi2 = df.LinearVariationalSolver(problem_chi2)
    solver_chi2.parameters["krylov_solver"]["absolute_tolerance"] = 1e-15

    solver_chi2.solve()

    with df.XDMFFile(mesh.mpi_comm(),
                     "chi_Pe0_b{}.xdmf".format(b)) as xdmff:
        xdmff.write(chi_)
    
    integral = (2*df.assemble(chi_.dx(0)*df.dx)/V_Omega
                + df.assemble(df.inner(df.grad(chi_),
                                       df.grad(chi_))*df.dx)/V_Omega)

    if rank == 0:
        print("b = {}, D_eff/D = {}".format(b, 1+integral))

    data[ib, 0] = b
    data[ib, 1] = 1+integral

if rank == 0:
    np.savetxt("{}/Pe0.dat".format(folder), data)
