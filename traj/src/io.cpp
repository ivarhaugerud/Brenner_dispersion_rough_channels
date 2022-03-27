#include <iostream>
#include <vector>
#include <map>
#include <boost/algorithm/string.hpp>
#include <fstream>
#include <filesystem>
#include "io.hpp"

using namespace std;

map<string, string> load_settings(string casefile){
  ifstream input(casefile);
  string line;
  string el;
  map<string, string> settings;
  
  while (getline(input, line)){
    size_t pos = line.find("=");

    // cout << line.substr(0, pos) << ": " << line.substr(pos+1) << endl;
    settings[line.substr(0, pos)] = line.substr(pos+1);
  }
  return settings;
}

vector<string> get_files(const string& s)
{
    vector<string> r;
    for(auto& p : filesystem::directory_iterator(s))
        if (!p.is_directory())
            r.push_back(p.path().string());
    return r;
}

vector<vector<string>> load_grid(string infile){
  ifstream input(infile);

  vector<char> x;
  string line;

  vector<vector<string>> sites;
  int nx = 0;
  while (getline(input, line)){
    // cout << line << endl;
    vector<string> sites_loc;
    boost::split(sites_loc, line, boost::is_any_of(" "));
    int sites_loc_size = sites_loc.size();
    nx = max(nx, sites_loc_size);
    sites.push_back(sites_loc);
  }
  int ny = sites.size();

  cout << "Size of grid: " << nx << " x " << ny << endl;
  for (int iy=0; iy < ny; ++iy){
    int nx_loc = sites[iy].size();
    for (int j=0; j < nx-nx_loc; ++j){
      sites[iy].push_back("s");
    }
  }
  return sites;
}

vector<vector<string>> load_fields(string infile){
  ifstream input(infile);

  vector<char> x;
  string line;

  vector<vector<string>> data;
  int nx = 0;
  while (getline(input, line)){
    // cout << line << endl;
    vector<string> data_loc;
    boost::split(data_loc, line, boost::is_any_of(" "));
    int data_loc_size = data_loc.size();
    nx = max(nx, data_loc_size);
    data.push_back(data_loc);
  }
  return data;
}

void print_grid(bool** grid, const int nx, const int ny){
  for (int iy=0; iy < ny; ++iy){
    for (int ix=0; ix < nx; ++ix){
      cout << grid[iy][ix];
    }
    cout << endl;
  }
}

void copy_arr(double*** f,
	      double*** f_prev,
	      bool** grid,
	      const int nx,
	      const int ny,
	      const int nc){
  for (int iy=0; iy < ny; ++iy){
    for (int ix=0; ix < nx; ++ix){
      if (grid[iy][ix]){
	for (int ic=0; ic < nc; ++ic){
	  f_prev[iy][ix][ic] = f[iy][ix][ic];
	}
      }
    }
  }
}

void copy_arr(double** a_from,
	      double** a_to,
	      bool** grid,
	      const int nx,
	      const int ny){
  for (int iy=0; iy < ny; ++iy){
    for (int ix=0; ix < nx; ++ix){
      if (grid[iy][ix]){
	a_to[iy][ix] = a_from[iy][ix];
      }
    }
  }
}

void dump2file(string filename,
	       double** rho,
	       double** m_x,
	       double** m_y,
	       bool** grid,
	       const int nx,
	       const int ny){
  ofstream outfile(filename);
  outfile.precision(17);
  for (int iy=0; iy < ny; ++iy){
    for (int ix=0; ix < nx; ++ix){
      if (grid[iy][ix]){
	outfile << ix << " " << iy << " " << rho[iy][ix] << " " << m_x[iy][ix] << " " << m_y[iy][ix] << endl;
      }
    }
  }
  outfile.close();
}
