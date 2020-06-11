#ifndef INPUT
#define INPUT

#include "analytic.hpp"

#include <string>
#include <vector>
#include <fstream>
#include <exception>

struct Tolerance_control
{
  int max_outer{100};
  double k{1e-6};
  double phi{1e-5};

  int max_inner{1000};
  double inner_rel{1e-12};
  double inner_abs{1e-24};
};

class Input
{
  public:
    Input(std::string fname);

    void open(std::string fname);
    void read_core();
    void read_state();
    void echo(std::ostream& os) const;

    class Invalid_input_card : std::exception
    {
      public:
        Invalid_input_card(std::string str)
        {
          msg = "Invalid input card: " + str;
        }
        const char* what() const noexcept override
        {
          return msg.c_str();
        }
      private:
        std::string msg;
    };

    class Invalid_file_name : std::exception
    {
      public:
        Invalid_file_name(std::string str)
        {
          msg = "Invalid file name: " + str;
        }
        const char* what() const noexcept override
        {
          return msg.c_str();
        }
      private:
        std::string msg;
    };

    class Invalid_reread_input : std::exception
    {
      public:
        Invalid_reread_input()
        {
          msg = "Input cannot be reread.";
        }
        const char* what() const noexcept override
        {
          return msg.c_str();
        }
      private:
        std::string msg;
    };

    // CORE
    std::string mesh_file() const { return fname_mesh; }
    std::string xs_file() const { return fname_xs; }
    int refine_serial() const { return rs; }
    int refine_parallel() const { return rp; }
    std::vector<int> boundary_conditions() const { return bc;}
    Analytic::Reference reference() const { return ref; }

    // STATE
    bool verbose() const { return verb; }
    int max_inner() const { return tol.max_inner; }
    double inner_rel_tol() const { return tol.inner_rel; }
    double inner_abs_tol() const { return tol.inner_abs; }
    int max_outer() const { return tol.max_outer; }
    double k_tol() const { return tol.k; }
    double phi_tol() const { return tol.phi; }

  private:
    static constexpr char comment_char{'#'};
    bool read_core_done{false};
    std::ifstream ifs;

    std::string stub;

    // CORE
    std::string fname_mesh; // mesh_file
    std::string fname_xs; // xs_file
    std::vector<int> bc; // boundary_conditions
    int rs; // refine_serial
    int rp; // refine_parallel
    Analytic::Reference ref{Analytic::invalid}; // reference problem

    // STATE
    bool verb; // verbose
    Tolerance_control tol;

    std::string path_stub(std::string path);

    static void reset_state(std::ifstream& ifs);
    static bool read_bool(std::ifstream& ifs);
    static void read_vector(std::ifstream& ifs, std::vector<int>& vec);
};

struct Arguments
{
  Arguments(const Input& inp);

  const char* mesh_file{nullptr};
  const char* device_config{nullptr};

  int order;

  bool verbose;
  bool static_cond; // static condensation
  bool pa; // partial assembly
};

#endif
