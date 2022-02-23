#ifndef XSLIBRARY
#define XSLIBRARY

#include "hdf5_access_wrapper.hpp"
#include "mfem.hpp"

#include <exception>
#include <fstream>
#include <string>
#include <vector>

struct XSRecord
{
  XSRecord() {}
  XSRecord(int ngroup) { set_ngroup(ngroup); }
  XSRecord(int ngroup, int nmoment) { set_ngroup(ngroup, nmoment); }

  void set_ngroup(int ngroup, int nmoment=1);
  int ngroup() const { return G; };
  int nmoment() const { return N; };

  void calculate_removal_xs();
  void self_scatter_rebalance();
  void evaluate_upscatter();

  std::string name;
  int zaid{0};
  double nden{0.0};
  bool is_fiss{false};

  int G{0};
  int N{0};

  std::vector<double> sigma_tr;
  std::vector<double> sigma_t;
  mfem::DenseMatrix sigma_r;
  std::vector<double> nusf, kappasf;
  std::vector<double> chi;
  std::vector<double> source; // fixed source

  std::vector<bool> is_upscatter; // is the upscattering into this group?

  mfem::SparseMatrix sigma_s;
};

class XSLibrary
{
public:
  int ngroup() const { return G; }
  int nmoment() const { return M; }
  int niso() const { return N; }

  void read_userxs(const char* const fname);
  void read_userxs(const std::string& fname);
  void read_hdf5xs(const char* const fname, const char* const lib_name);
  void read_hdf5xs(const std::string& fname, const std::string& lib_name);

  void Print(std::ostream& os);
  void calc_tcp0();

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
  int M{1}; // number of removal cross section moments present

  void initialize_records();

  void read_hdf5xs_isotxs(Hdf5_access_wrapper& acc,
                          const std::string& lib_name);

  static constexpr char comment_char{'#'};
  static constexpr int hollerith_length{8};

  template <typename T>
  static void read_vector(std::istream& is, std::vector<T>& vec)
  {
    for (T& val : vec) is >> val;
  }

  template <typename T>
  static void write_vector(std::ostream& os, const std::vector<T>& vec)
  {
    for (T val : vec) os << val << '\n';
  }
};

#endif
