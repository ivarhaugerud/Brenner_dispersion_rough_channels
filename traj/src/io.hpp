#include <iostream>
#include <vector>
#include <map>

#ifndef __IO_HPP
#define __IO_HPP

using namespace std;

map<string, string> load_settings(string casefile);
vector<string> get_files(const string& s);
vector<vector<string>> load_grid(string infile);
vector<vector<string>> load_fields(string infile);
void print_grid(bool** grid, const int nx, const int ny);
void copy_arr(double*** f,
	      double*** f_prev,
	      bool** grid,
	      const int nx,
	      const int ny,
	      const int nc);
void copy_arr(double** a_from,
	      double** a_to,
	      bool** grid,
	      const int nx,
	      const int ny);
void dump2file(string filename,
	       double** rho,
	       double** m_x,
	       double** m_y,
	       bool** grid,
	       const int nx,
	       const int ny);

#endif
