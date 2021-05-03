#ifndef INPUT
#define INPUT

#include "analytic.hpp"
#include "mfem.hpp" // Array<int>
#include "xslibrary.hpp"
#include "solver_choice.hpp"
#include "quadrature.hpp"

#include <exception>
#include <fstream>
#include <string>
#include <vector>

struct Tolerance_control
{
  int max_outer{100};
  double k{1e-6};
  double phi{1e-5};

  int max_inner{1000};
  double inner_rel{1e-8};
  double inner_abs{1e-10};

  int temp_max_outer{100};
  double temp{1.0};
  int temp_max_inner{1000};
  double temp_inner_rel{1e-6};
  double temp_inner_abs{1e-6};
};

struct Input
{
public:
  Input(std::string fname);
  Input(const char* fname);

  void open();
  void read_core();
  void read_state();
  int nstate();
  void reconcile_enums();
  void echo(std::ostream& os) const;

  int hypre_print_level() const { return verbose ? 1 : 0; };
  int mfem_print_level() const { return verbose ? 1 : 0; };
  int spn_eq() const { return (spn + 1) / 2; }
  int spn_order() const { return spn; }

  class Invalid_input_card : std::exception
  {
  public:
    Invalid_input_card(std::string str) : msg{"Invalid input card: " + str} {}
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

  class Invalid_reread_input : std::exception
  {
  public:
    Invalid_reread_input() { msg = "Input cannot be reread."; }
    const char* what() const noexcept override { return msg.c_str(); }

  private:
    std::string msg;
  };

  // general
  const char* input_file{""};

  // CORE
  const char* mesh_file{""};
  mfem::Array<int> boundary_conditions;
  int order{1};
  int refine_serial{0};
  int refine_parallel{0};
  const char* reference{"invalid"};
  const char* device{"cpu"};
  bool static_condensation{false};
  bool verbose{false};
  std::string xs_file;
  std::string hdf5xs_lib_name;

  // actual enum value
  Analytic::Reference ref{Analytic::invalid};
  XSLibrary::Format xs_format{XSLibrary::invalid};

  // STATE
  Tolerance_control tol;
  const char* solver{"lobpcg"};
  bool flux_free_chi{false};
  int spn{1};

  const char* quadrature{"level_symmetric"};
  int quad_legendre_order{0};
  int quad_chebyschev_order{0};
  int quad_order{2};
  Quadrature_choice quadrature_type{Quadrature_choice::level_symmetric};

  // THERMAL
  double rxpwr{1.0}; // reactor power [W]
  double temp_boundary{273.15}; // boundary temperature [K]
  double temp_heat_pipe{300.}; // heat pipe constant temperature [K]

  // GPU
  int ndevices{0}; // If (ndevices>0), use MPITeams. Else, use MPIExclusive.

  Solver_choice sol{Solver_choice::lobpcg};

  // this is public so that other blocks (e.g. MATERIAL) can be read
  std::ifstream ifs;

  std::string concatenate_path(const std::string& fname) const;
  std::string input_file_stub();

private:

  // general
  static constexpr char comment_char{'#'};
  int number_of_state{0};

  // strings corresponding to const char*
  std::string input_file_str;
  std::string input_file_stub_str;
  // [CORE]
  std::string mesh_file_str;
  std::string device_str;
  std::string reference_str;
  // [STATE]
  std::string solver_str;
  std::string quadrature_str;

  std::string stub;
  std::string path_stub(std::string path);

  static void reset_block(std::ifstream& ifs, const std::string& block_head);
  static bool read_bool(std::ifstream& ifs);
  static void read_vector(std::ifstream& ifs, mfem::Array<int>& vec);

};

mfem::OptionsParser custom_arguments(Input& inp, int argc, char* argv[]);

#endif
