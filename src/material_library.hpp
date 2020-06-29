#ifndef MATERIAL_LIBRARY
#define MATERIAL_LIBRARY

#include "xslibrary.hpp"

#include <vector>
#include <string>
#include <iostream>
#include <regex>

struct Material_record
{
  // TODO this should probably be its own struct and a single vector
  std::vector<double> nden;
  std::vector<std::string> iso_name;
  std::vector<int> iso_idx;
  std::vector<std::reference_wrapper<XSRecord>> iso_rec;

  std::string name;
  int niso;
  int G;

  // utility functions

  int ngroup() { return G; }

  // calculate chi for all energy groups
  // phi_g is the flux spectra at a given position
  std::vector<double> calculate_chi(const std::vector<double>& phi_g);

  // calculate chi for a single energy group
  // phi_g is the flux spectra at a given position
  double calculate_chi(const std::vector<double>& phi_g, int g);
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
        {}
        const char * what() const noexcept override { return msg.c_str(); }
      private:
        const std::string msg;
    };

  private:
    const XSLibrary& xs; // associated microscopic cross section library
    int G; // set in constructor
    int nMat{0};
    std::vector<Material_record> mat;

    static const std::regex block_head_regex_format;
    static void reset_block(std::ifstream& ifs, const std::string& block_head);
    static constexpr char comment_char{'#'};
};

#endif
