#ifndef ANALYTIC
#define ANALYTIC

#include "mfem.hpp"
#include "result.hpp"
#include "xslibrary.hpp"

#include "solver_choice.hpp"

#include <functional>

namespace Analytic_solutions
{
  extern const XSLibrary* xs_global;

  // Zeroth zero of the first Bessel function of the first kind.
  static constexpr double alpha0{2.404825557695772768622};

  // TODO these will have to be rewritten to be VectorFunctionCoefficient
  double fixed_src_fun(const mfem::Vector&);
  double saaf_benchmark_fun(const mfem::Vector&);
  double oned_oneg_fun(const mfem::Vector&);
  double twod_oneg_fun(const mfem::Vector&);
  double threed_oneg_fun(const mfem::Vector&);
  double finite_cyl_fun(const mfem::Vector&);
  double tworeg_fun(const mfem::Vector&);

  double oned_oneg_keff();
  double twod_oneg_keff();
  double threed_oneg_keff();
  double finite_cyl_keff();
  double tworeg_keff();
  double oned_twog_keff();
  double oned_oneg_spn_keff();

  double oned_twog_ratio();
  double oned_oneg_spn_ratio();

} // namespace Analytic_solutions

class Analytic
{
public:
  class Invalid_reference_name
  {
  };

  enum Reference
  {
    invalid = -1,
    fixed_src,
    saaf_benchmark,
    oned_oneg,
    twod_oneg,
    threed_oneg,
    square,
    finite_cyl,
    tworeg,
    oned_twog,
    oned_oneg_spn
  };

  Analytic(const Reference ref, const Solver_choice sol, Result& res, 
           const XSLibrary& xs,
           const mfem::Mesh* const mesh);

  void analyze(std::ostream& os) const;
  void print(std::ostream& os) const;

  // helper functions for dealing with Reference enum
  static Reference reference_str_to_enum(std::string str);
  static std::string reference_enum_to_str(Reference ref);

  ~Analytic();

private:
  void normalize_solution_flux() const;

  const Reference reference_solution;
  const Solver_choice solver;
  int nVert;
  int nDim;
  double keff;

  std::vector<mfem::Vector> node;
  mfem::GridFunction* phi{nullptr};
  mfem::Vector* phi2{nullptr};

  using mfem_function = double(const mfem::Vector&);
  using mfem_vector_function = void(const mfem::Vector&, mfem::Vector&);
  using keff_function = double();
  mfem_function* fun;
  keff_function* analytic_keff;
};

#endif
