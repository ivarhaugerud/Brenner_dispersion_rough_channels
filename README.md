# General 
The code is divded into two main sections. In the main directory one solves the Navier--Stokes equations and Brenner equation to calculate the effective diffusion coefficient by running ```flow.py``` which uses the other files in the directory. To use Brenners theory without flow, the file ```pure_diff.py``` can be ran. In the sub-folder traj, one finds the C++ code to simulate advected (or not) random walkers. To find the neccesary flow field, one first needs to run ```flow.py```.

# Requirements 

To run the python 3 code, one needs the packages dolfin, h5py, meshpy, numpy and matplotlib.
Dolfin/FEniCS is used to solve the equations using the finite element method. Its documentation can be found at [https://fenicsproject.org/olddocs/dolfin/latest/python/index.html]

# Rough-tracers

Example use:

Navier-Stokes + Brenner's solution
```
python3 flow.py -logPe_min 0 -logPe_max 4 -logPe_N 5 -R 0.0 -res 100 -b 0.2
```

Pure diffusion with Brenners solution
```
python3 pure_diff.py -res 100 -b_min 0.0 -b_max 1.9 -b_N 20
```

# Traj (particle tracking) is a C++ layer that uses the flow field produced in the preceding step

To compile
```
cd traj
cmake .
make
```

To run
```
./traj ../data_square/flow_Re0.0_b0.2.h5 Dm=1.0 T=100.0 dt=0.001 Nrw=100 traj_intv=0.1 pos_intv=0.1 stat_intv=0.1 dump_traj=true verbose=true U=1.0
```
to plot:
```
gnuplot
pl 'data_square/RandomWalkers/Re0.000000_b0.200000_Dm1.000000_U1.000000_dt0.001000_Nrw100/tdata.dat' u 1:3 w l
```
or for trajectories
```
pl 'data_square/RandomWalkers/Re0.000000_b0.200000_Dm1.000000__dt0.001000_Nrw100/Trajectories/traj_0.traj' u 2:3 w l
```
