#ifndef XSLIBRARY
#define XSLIBRARY

#include <exception>
#include <string>
#include <vector>
#include <fstream>

#include "mfem.hpp"

struct XSRecord
{
  XSRecord() {}
  XSRecord(int ngroup) { set_ngroup(ngroup); }

  void set_ngroup(int ngroup);
  int ngroup() const { return G; };

  void calculate_removal_xs();
  void self_scatter_rebalance();
  void evaluate_upscatter();

  std::string name;
  int zaid{0};
  double nden{0.0};
  bool is_fiss{false};

  int G{0};

  std::vector<double> sigma_tr;
  std::vector<double> sigma_t;
  std::vector<double> sigma_r;
  std::vector<double> nusf;
  std::vector<double> chi;
  std::vector<double> source; // fixed source

  std::vector<bool> is_upscatter; // is the upscattering into this group?

  mfem::SparseMatrix sigma_s;
};

class XSLibrary
{
public:
  int ngroup() const { return G; }
  int niso() const { return N; }

  void read_userxs(const std::string& fname);
  void read_userxs(const char* fname);
  void Print(std::ostream& os);

  class Invalid_input_card : std::exception
  {
  public:
    Invalid_input_card(std::string str) { msg = "Invalid input card: " + str; }
    const char* what() const noexcept override { return msg.c_str(); }

  private:
    std::string msg;
  };

  class Invalid_file_name : std::exception
  {
  public:
    Invalid_file_name(std::string str) { msg = "Invalid file name: " + str; }
    const char* what() const noexcept override { return msg.c_str(); }

  private:
    std::string msg;
  };

  std::vector<XSRecord> rec;

private:
  int G{0}; // number of energy groups
  int N{0}; // number of materials/isotopes present

  void initialize_records();

  static constexpr char comment_char{'#'};

  template<typename T>
  static void read_vector(std::istream& is, std::vector<T>& vec)
  {
    for (T& val : vec)
      is >> val;
  }

  template<typename T>
  static void write_vector(std::ostream& os, const std::vector<T>& vec)
  {
    for (T val : vec)
      os << val << '\n';
  }

};

#endif
