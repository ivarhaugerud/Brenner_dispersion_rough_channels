#include <iostream>
#include <dolfin.h>
#include "io.hpp"
#include <vector>
#include <filesystem>
#include <boost/algorithm/string.hpp>
#include "Velocity.h"
#include <fstream>
#include <random>

using namespace dolfin;

string create_folder(string folder){
  if (!std::filesystem::is_directory(folder)){
    std::filesystem::create_directory(folder);
  }
  return folder;
}

bool eval_field(double &ux, double &uy, std::shared_ptr<Mesh> mesh, std::shared_ptr<Function> u_,
		const double x, const double y){
  Array<double> vals(2);
  Array<double> xarr(2);

  xarr[0] = x;
  xarr[1] = y;
  
  const double* _x = xarr.data();
  const Point point(mesh->geometry().dim(), _x);

  // index of first cell containing point
  unsigned int id
    = mesh->bounding_box_tree()->compute_first_entity_collision(point);

  if (id == std::numeric_limits<unsigned int>::max()){
    return false;
  }
  const Cell cell(*mesh, id);
  ufc::cell ufc_cell;
  cell.get_cell_data(ufc_cell);
  
  u_->eval(vals, xarr, cell, ufc_cell);
  
  ux = vals[0];
  uy = vals[1];
  return true;
}

double modulox(const double x, const double L){
  if (x > 0){
    return fmod(x, L);
  }
  else {
    return fmod(x, L)+L;
  }
}

void print_param(string key, double val){
  std::cout << key << " = " << val << std::endl;
}

void print_param(string key, int val){
  std::cout << key << " = " << val << std::endl;
}

void print_param(string key, string val){
  std::cout << key << " = " << val << std::endl;
}

int main(int argc, char* argv[]){
  // Assert no parallel
  if (dolfin::MPI::size(MPI_COMM_WORLD) > 1)
    std::cout << "No parallel" << std::endl;

  // Input parameters
  if (argc < 2) {
    std::cout << "Specify an input HDF5 file." << std::endl;
    return 0;
  }
  // Default parameters
  double Dm = 1.0;
  double T = 100.;
  int Nrw = 10;
  double traj_intv = 0.01;
  double pos_intv = 1.0;
  double stat_intv = 1.0;
  bool dump_traj = true;
  bool verbose = false;
  double U0 = 1.0;
  double dt = 0.01;

  size_t found;
  string argstr, key, val;
  for (int iarg=2; iarg < argc; ++iarg){
    argstr = argv[iarg];
    found = argstr.find('=');
    if (found != string::npos){
      key = argstr.substr(0, found);
      val = argstr.substr(found+1);
      boost::trim(key);
      boost::trim(val);

      if (key == "Dm") Dm = stod(val);
      if (key == "T") T = stod(val);
      if (key == "dt") dt = stod(val);
      if (key == "Nrw") Nrw = stoi(val);
      if (key == "traj_intv") traj_intv = stod(val);
      if (key == "pos_intv") pos_intv = stod(val);
      if (key == "stat_intv") stat_intv = stod(val);
      if (key == "dump_traj") dump_traj = (val == "true" || val == "True") ? true : false;
      if (key == "verbose") verbose = (val == "true" || val == "True") ? true : false;
      if (key == "U") U0 = stod(val);
    }
  }

  if (verbose){
    print_param("Dm       ", Dm);
    print_param("T        ", T);
    print_param("dt       ", dt);
    print_param("Nrw      ", Nrw);
    print_param("traj_intv", traj_intv);
    print_param("pos_intv ", pos_intv);
    print_param("stat_intv", stat_intv);
    print_param("dump_traj", dump_traj ? "true" : "false");
    print_param("verbose  ", verbose ? "true" : "false");
    print_param("U        ", U0);
  }

  string h5file = string(argv[1]);

  if (!filesystem::exists(h5file)){
    std::cout << "No such file: " << h5file << std::endl;
    exit(0);
  }
  
  std::size_t botDirPos = h5file.find_last_of("/");
  std::size_t extPos = h5file.find_last_of(".");

  std::string folder = h5file.substr(0, botDirPos);
  std::string filename = h5file.substr(botDirPos+1, extPos-botDirPos-1);

  // std::cout << folder << " " << filename << std::endl;
  
  std::size_t splitterPos = filename.find_last_of("_");
  std::string argRe = filename.substr(5, splitterPos-2);
  std::string argb = filename.substr(splitterPos+1, filename.length());
  
  double Re = stod(argRe.substr(2, argRe.length()));
  double b = stod(argb.substr(1, argb.length()));
  
  // std::cout << Re << " " << b << std::endl;
  
  string rwfolder = create_folder(folder + "/RandomWalkers/");
  string newfolder = create_folder(rwfolder +
    "/Re" + to_string(Re) + "_b" + to_string(b) +
    "_Dm" + to_string(Dm) + "_U" + to_string(U0) +
    "_dt" + to_string(dt) + "_Nrw" + to_string(Nrw) + "/");
  string trajfolder = create_folder(newfolder + "Trajectories/");
  string posfolder = create_folder(newfolder + "Positions/");

  // Mesh and stuff
  Mesh mesh;
  
  dolfin::HDF5File h5f = dolfin::HDF5File(MPI_COMM_WORLD,
			  // "../data_square/flow_Re0.0_b0.3.h5",
			  h5file, "r");

  h5f.read(mesh, "mesh", false);

  vector<double> xxx = mesh.coordinates();
  double x_min = xxx[0];
  double x_max = xxx[0];
  double y_min = xxx[1];
  double y_max = xxx[1];
  for (int it = 0; it < xxx.size(); it += 2){
    x_min = min(x_min, xxx[it]);
    x_max = max(x_max, xxx[it]);
    y_min = min(y_min, xxx[it+1]);
    y_max = max(y_max, xxx[it+1]);
  }

  if (abs(x_min) > 1e-9 ||
      abs(y_min-(-1-b/2)) > 1e-9 ||
      abs(y_max-(1+b/2)) > 1e-9) {
    std::cout << "Mesh not compatible..." << std::endl;
    std::cout << "x: " << x_min << " " << x_max << std::endl;
    std::cout << "y: " << y_min << " " << y_max << std::endl;
    exit(0);
  }
  double domain_width = x_max;
  if (verbose){
    print_param("domain_width", domain_width);
    print_param("b", b);
  }
  
  auto mesh2 = std::make_shared<Mesh>(mesh);
  
  MeshFunction<std::size_t> subd(mesh2, 1);
  
  h5f.read(subd, "subd");

  auto V = std::make_shared<Velocity::FunctionSpace>(mesh2);
  auto u_ = std::make_shared<Function>(V);

  h5f.read(*u_, "U");

  XDMFFile("test_out.xdmf").write(*u_);

  random_device rd;
  mt19937 gen(rd());

  uniform_real_distribution<> uni_dist_y(-1+b/2, 1-b/2);
  normal_distribution<double> rnd_normal(0.0, 1.0);

  double x, y;
  double ux, uy;
  bool in_domain;
  double* x_rw = new double[Nrw];
  double* y_rw = new double[Nrw];
  double* ux_rw = new double[Nrw];
  double* uy_rw = new double[Nrw];

  ofstream* traj_outs = new ofstream[Nrw];
  for (int irw=0; irw < Nrw; ++irw){
    // Initial position
    if (dump_traj){
      string filename = trajfolder + "traj_" + to_string(irw) + ".traj";
      traj_outs[irw].open(filename, ofstream::out);
      if (verbose)
	std::cout << "Opened: " << irw << std::endl;
    }
    
    x = b;
    do {
      y = uni_dist_y(gen);
      in_domain = eval_field(ux, uy, mesh2, u_, x, y);
    } while (!in_domain);
    x_rw[irw] = x;
    y_rw[irw] = y;
    ux_rw[irw] = U0*ux;
    uy_rw[irw] = U0*uy;
  }

  double eta1, eta2;
  double dx_rw, dy_rw;
  int it = 0;

  double sqrt2Dmdt = sqrt(2*Dm*dt);
  if (verbose){
    print_param("sqrt(2*Dm*dt)", sqrt2Dmdt);
    print_param("U*dt", U0*dt);
    print_param("dx", mesh.hmax());
  }
  
  double x_mean, dx2_mean;
  double t = 0.;

  int n_accepted = 0;
  int n_declined = 0;
  
  ofstream statfile(newfolder + "/tdata.dat");
  ofstream declinedfile(newfolder + "/declinedpos.dat");
  while (t <= T){
    //
    if (it % int(pos_intv/dt) == 0){
      string posfile = posfolder + "xy_t" + to_string(t) + ".pos";
      ofstream pos_out(posfile);

      for (int irw=0; irw < Nrw; ++irw){
	pos_out << irw << " " << x_rw[irw] << " " << y_rw[irw] << std::endl;
      }
      pos_out.close();
    }

    if (it % int(stat_intv/dt) == 0){
      x_mean = 0.;
      dx2_mean = 0.;
      for (int irw=0; irw < Nrw; ++irw){
	x_mean += x_rw[irw]/Nrw;  // Sample mean
      }
      for (int irw=0; irw < Nrw; ++irw){
	dx2_mean += pow(x_rw[irw]-x_mean, 2)/(Nrw-1);  // Sample variance
      }
      statfile << t << " " << x_mean << " " << dx2_mean <<
	" " << n_accepted << " " << n_declined << std::endl;
    }

    for (int irw=0; irw < Nrw; ++irw){
      if (dump_traj && it % int(traj_intv/dt) == 0){
	traj_outs[irw] << t << " " << x_rw[irw] << " " << y_rw[irw]
		       << " " << ux_rw[irw] << " " << uy_rw[irw] << std::endl;
	// std::cout << "Wrote traj: " << irw << " of " << Nrw << std::endl;
      }
      dx_rw = ux_rw[irw]*dt;
      dy_rw = uy_rw[irw]*dt;
      if (Dm > 0.0){
	eta1 = rnd_normal(gen);
	eta2 = rnd_normal(gen);
	// std::cout << eta1 << " " << eta2 << std::endl;
	dx_rw += sqrt2Dmdt*eta1;
	dy_rw += sqrt2Dmdt*eta2;
      }
      in_domain = eval_field(ux, uy, mesh2, u_,
			     modulox(x_rw[irw]+dx_rw, domain_width),
			     y_rw[irw]+dy_rw);
      if (in_domain){
	x_rw[irw] += dx_rw;
	y_rw[irw] += dy_rw;
	ux_rw[irw] = U0*ux;
	uy_rw[irw] = U0*uy;
	n_accepted++;
      }
      else {
	n_declined++;
	declinedfile << t << " " << x_rw[irw]+dx_rw << " "
		     << y_rw[irw]+dy_rw << std::endl;
      }
    }
    t += dt;
    it += 1;
  }

  if (dump_traj){
    for (int irw=0; irw < Nrw; ++irw){
      traj_outs[irw].close();
    }
  }
  statfile.close();
  declinedfile.close();

  return 0;
}
