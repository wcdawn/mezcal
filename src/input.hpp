#ifndef INPUT
#define INPUT

#include "analytic.hpp"
#include "mfem.hpp" // Array<int>

#include <exception>
#include <fstream>
#include <string>
#include <vector>

#include<regex>

struct Tolerance_control
{
  int max_outer{100};
  double k{1e-6};
  double phi{1e-5};

  int max_inner{1000};
  double inner_rel{1e-12};
  double inner_abs{1e-24};
};

struct Input
{
public:
  Input(std::string fname);
  Input(const char* fname);

  void open();
  void read_core();
  void read_state();
  void reconcile_enums();
  void echo(std::ostream& os) const;

  int hypre_print_level() const { return verbose ? 1 : 0; };
  int mfem_print_level() const { return verbose ? 1 : 0; };

  class Invalid_input_card : std::exception
  {
  public:
    Invalid_input_card(std::string str) : msg {"Invalid input card: " + str} {}
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
  const char* input_file;

  // CORE
  const char* mesh_file{""};
  mfem::Array<int> boundary_conditions;
  int order{1};
  int refine_serial{0};
  int refine_parallel{0};
  const char* xs_file{""};
  const char* reference{"invalid"};
  const char* device{"cpu"};
  bool static_condensation{false};

  // actual enum value
  Analytic::Reference ref{Analytic::invalid};

  // STATE
  bool verbose{false};
  Tolerance_control tol;
  const char* solver{"lobpcg"};

  enum Solver_choice
  {
    invalid = -1,
    power_iteration,
    lobpcg
  };

  Solver_choice sol{lobpcg};

private:
  static constexpr char comment_char{'#'};
  std::ifstream ifs;

  // strings corresponding to const char*
  // [CORE]
  std::string input_file_str;
  std::string mesh_file_str;
  std::string xs_file_str;
  std::string device_str;
  std::string reference_str;
  // [STATE]
  std::string solver_str;

  std::string stub;
  std::string path_stub(std::string path);

  static const std::regex block_head_regex_format;
  static void reset_block(std::ifstream& ifs, const std::string& block_head);
  static bool read_bool(std::ifstream& ifs);
  static void read_vector(std::ifstream& ifs, mfem::Array<int>& vec);

  static std::string solver_choice_enum_to_str(const Solver_choice& solver);
  static Solver_choice solver_choice_str_to_enum(const std::string& solver);
};

mfem::OptionsParser custom_arguments(Input& inp, int argc, char* argv[]);

#endif
