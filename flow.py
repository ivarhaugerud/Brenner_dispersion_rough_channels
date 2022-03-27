import dolfin as df
import argparse
import matplotlib.pyplot as plt
import numpy as np
from mesh_box import square_rough_mesh
import os
from bcs import PeriodicBC, Wall, NotWall
from mpi4py import MPI

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
folder = "data_square"


# Form compiler options
df.parameters["form_compiler"]["optimize"] = True
df.parameters["form_compiler"]["cpp_optimize"] = True

parser = argparse.ArgumentParser(description="Taylor dispersion")
parser.add_argument("-res", type=int, default=32, help="Resolution")
parser.add_argument("-R", type=float, default=0.0,
                    help="Imposed Reynolds number")
parser.add_argument("-logPe_min", type=float, default=0.0, help="Min Pe")
parser.add_argument("-logPe_max", type=float, default=5.0, help="Max Pe")
parser.add_argument("-logPe_N", type=int, default=10, help="Num Pe")
parser.add_argument("-b", type=float, default=0.5, help="Roughness b")
parser.add_argument("--onlyflow", action="store_true", help="only flow")
args = parser.parse_args()

N = args.res
R = df.Constant(args.R)

Ly = 2.0

dx = Ly/N

# Change this
# mesh = df.UnitSquareMesh(N, N)
mesh = square_rough_mesh(args.b, dx)
coords = mesh.coordinates()[:]
# coords[:, 1] *= Ly
Lx = df.MPI.max(df.MPI.comm_world, coords[:, 0].max())

Eu = df.VectorElement("Lagrange", mesh.ufl_cell(), 3)
Ep = df.FiniteElement("Lagrange", mesh.ufl_cell(), 2)
Echi = df.FiniteElement("Lagrange", mesh.ufl_cell(), 2)

pbc = PeriodicBC(Lx, Ly)
wall = Wall(Lx, Ly)
notwall = NotWall(Lx, Ly)

subd = df.MeshFunction("size_t", mesh, mesh.topology().dim()-1)
subd.set_all(0)
wall.mark(subd, 1)
notwall.mark(subd, 0)

if rank == 0 and not os.path.exists(folder):
    os.makedirs(folder)

with df.XDMFFile(mesh.mpi_comm(), "{}/subd_b{}.xdmf".format(
        folder, args.b)) as xdmff:
    xdmff.write(subd)

W = df.FunctionSpace(mesh, df.MixedElement([Eu, Ep]),
                     constrained_domain=pbc)
S = df.FunctionSpace(mesh, Echi, constrained_domain=pbc)

w_ = df.Function(W)
u_, p_ = df.split(w_)
w = df.TrialFunction(W)
u, p = df.split(w)
v, q = df.TestFunctions(W)

f = df.Constant((1., 0.))

F = (
    R*df.inner(df.grad(u_)*u_, v)*df.dx
    + df.inner(df.grad(u_), df.grad(v))*df.dx
    - df.div(v)*p_*df.dx - df.div(u_)*q*df.dx
    - df.dot(f, v)*df.dx
)

J = df.derivative(F, w_, du=w)

bcu = df.DirichletBC(W.sub(0), df.Constant((0., 0.)),
                     subd, 1)

x0, y0 = coords[0, 0], coords[0, 1]
x0 = comm.bcast(x0, root=0)
y0 = comm.bcast(y0, root=0)
# distribute!

bcp = df.DirichletBC(W.sub(1), df.Constant(0.),
                     ("abs(x[0]-({x0})) < DOLFIN_EPS && "
                      "abs(x[1]-({y0})) < DOLFIN_EPS").format(x0=x0, y0=y0),
                     "pointwise")

bcs = [bcu, bcp]

problem = df.NonlinearVariationalProblem(F, w_, bcs=bcs, J=J)
solver = df.NonlinearVariationalSolver(problem)
solver.parameters["newton_solver"]["absolute_tolerance"] = 1e-14
solver.parameters["newton_solver"]["krylov_solver"]["absolute_tolerance"] = 1e-14
solver.parameters["newton_solver"]["krylov_solver"]["nonzero_initial_guess"] = True

solver.solve()

one = df.interpolate(df.Constant(1.), S)
V_Omega = df.assemble(one*df.dx)

ux_mean = df.assemble(u_[0]*df.dx)/V_Omega

if rank == 0:
    print("ux_mean = {}".format(ux_mean))

Re = args.R*ux_mean
if rank == 0:
    print("Re = {}".format(Re))

# Part 2: chi
U_, P_ = w_.split(deepcopy=True)
U_.rename("u", "tmp")
U_.vector()[:] /= ux_mean

with df.XDMFFile(mesh.mpi_comm(), "{}/U_Re{}_b{}.xdmf".format(
        folder, Re, args.b)) as xdmff:
    xdmff.write(U_)

with df.HDF5File(mesh.mpi_comm(), "{}/flow_Re{}_b{}.h5".format(
        folder, Re, args.b), "w") as h5f:
    h5f.write(mesh, "mesh")
    h5f.write(U_, "U")
    h5f.write(subd, "subd")

with df.HDF5File(mesh.mpi_comm(), "{}/up_Re{}_b{}.h5".format(folder, Re, args.b), "w") as h5f:
    h5f.write(U_, "u")
    h5f.write(P_, "p")

with df.HDF5File(mesh.mpi_comm(), "{}/mesh2d.h5".format(folder), "w") as h5f:
    h5f.write(mesh, "mesh")

if args.onlyflow:
    exit("Only flow!")

n = df.FacetNormal(mesh)

Pe = df.Constant(1.0)

chi = df.TrialFunction(S)
chi_ = df.Function(S)
psi = df.TestFunction(S)

ds = df.Measure("ds", domain=mesh, subdomain_data=subd)

F_chi = (n[0]*psi*ds(1)
         + df.inner(df.grad(chi), df.grad(psi))*df.dx
         + Pe*psi*df.dot(U_, df.grad(chi))*df.dx
         + Pe*(U_[0] - df.Constant(1.))*psi*df.dx)

a_chi, L_chi = df.lhs(F_chi), df.rhs(F_chi)

solver_chi = df.PETScKrylovSolver("gmres")
nullvec = df.Vector(chi_.vector())
S.dofmap().set(nullvec, 1.0)
nullvec *= 1.0/nullvec.norm("l2")
nullspace = df.VectorSpaceBasis([nullvec])

problem_chi2 = df.LinearVariationalProblem(a_chi, L_chi, chi_, bcs=[])
solver_chi2 = df.LinearVariationalSolver(problem_chi2)

solver_chi2.parameters["krylov_solver"]["absolute_tolerance"] = 1e-15

logPe_arr = np.linspace(args.logPe_min, args.logPe_max, args.logPe_N)

if rank == 0:
    data = np.zeros((len(logPe_arr), 3))

for iPe, logPe in enumerate(logPe_arr):
    Pe_loc = 10**logPe
    Pe.assign(Pe_loc)

    solver_chi2.solve()

    integral = (2*df.assemble(chi_.dx(0)*df.dx)/V_Omega
                + df.assemble(df.inner(df.grad(chi_),
                                       df.grad(chi_))*df.dx)/V_Omega)

    if rank == 0:
        print("Pe = {}, D_eff/D = {}".format(Pe_loc, 1+integral))

        data[iPe, 0] = Pe_loc
        data[iPe, 1] = 1+integral
        data[iPe, 2] = integral/Pe_loc**2

if rank == 0:
    np.savetxt("{}/Re{}_b{}.dat".format(folder, Re, args.b), data)
