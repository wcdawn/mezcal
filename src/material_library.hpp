#ifndef MATERIAL_LIBRARY
#define MATERIAL_LIBRARY

#include "mfem.hpp"
#include "xslibrary.hpp"

#include <iostream>
#include <string>
#include <vector>

struct Material_isotope
{
  std::string name;
  double nden;
  int idx;
  const XSRecord& rec;
};

struct Material_record
{
  std::vector<Material_isotope> iso;
  XSRecord rec;

  std::string name;
  int niso;
  int G;
  int N;

  int ngroup() const { return G; }
  int nmoment() const { return N; }

  // utility functions

  // This could be implemented as an overload but I want to make it explicitly
  // clear how chi is being collapsed.

  // calculate for all groups
  // phi_g is the flux spectra at a given position
  std::vector<double> calculate_chi(const std::vector<double>& phi_g) const;
  // alternatively, calculate chi weighting with only nusf
  std::vector<double> flux_free_chi() const;

  // calculate for a single group
  // These are less efficient but will be necessary to properly evaluate chi for
  // a given flux spectra. The main efficiency penalty will be a bigger problem
  // for many-group problems.
  // phi_g is the flux spectra at a given position
  double calculate_chi(const std::vector<double>& phi_g, int g) const;
  // alternatively, calculate chi weighting with only nusf
  double flux_free_chi(int g) const;

  // calculate macroscopic xs based on user-input number densities and
  // microscopic xs from XSLibrary
  void calculate_macroscopic_xs();

  enum Thermal_material
  {
    Na,
    Pb,
    PbBi,
    HT9,
    U10Zr,
    UN,
    UO2, // includes 5% porosity
  };

  Thermal_material k_mat{Thermal_material::HT9};

  static double thermal_conductivity(Thermal_material k_mat_in, double temperature);
  double thermal_conductivity(double temperature) const { return thermal_conductivity(k_mat, temperature); }

  double exact_area_volume{-1.0};

};

class Material_library
{
public:
  Material_library(const XSLibrary& xs);
  void read_material(std::ifstream& ifs);

  class Invalid_material_card : std::exception
  {
  public:
    Invalid_material_card(std::string str)
      : msg{"Inavlid material card: " + str}
    {
    }
    const char* what() const noexcept override { return msg.c_str(); }

  private:
    const std::string msg;
  };

  int ngroup() const { return G; }
  int nmoment() const { return M; }
  int nmat() const { return static_cast<int>(mat.size()); } 

  void set_nmoment(int m);

  std::vector<Material_record> mat;
  std::vector<double> chi;
  bool rx_wide_chi{false};

private:
  const XSLibrary& xs; // associated microscopic cross section library
  const int G;         // set in constructor
  int M; // non-const, can be updated/expanded with more moments for spn

  static void reset_block(std::ifstream& ifs, const std::string& block_head);
  static constexpr char comment_char{'#'};
};

#endif
