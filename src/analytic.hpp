#ifndef ANALYTIC
#define ANALYTIC

#include "mfem.hpp"
#include "result.hpp"
#include "xslibrary.hpp"

#include <functional>

namespace Analytic_solutions
{
  extern const XSLibrary* xs_global;

  // Zeroth zero of the first Bessel function of the first kind.
  static constexpr double alpha0{2.404825557695772768622};

  double fixed_src_fun(const mfem::Vector&);
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

}

class Analytic
{
public:
  class InvalidReferenceName
  {};

  enum Reference
  {
    invalid = -1,
    fixed_src,
    oned_oneg,
    twod_oneg,
    threed_oneg,
    square,
    finite_cyl,
    tworeg
  };

  Analytic(const Reference ref, Result& res, const XSLibrary& xs,
           const mfem::Mesh* const mesh);

  void analyze(std::ostream& os) const;
  void print(std::ostream& os) const;

  // helper functions for dealing with Reference enum
  static Reference reference_str_to_enum(std::string str);
  static std::string reference_enum_to_str(Reference ref);

private:

  void normalize_solution_flux() const;

  Reference reference_solution;
  int nVert;
  int nDim;
  double keff;
  const XSLibrary* const xs;

  std::vector<mfem::Vertex> node;
  mfem::GridFunction* phi{nullptr};

  typedef double mfem_function(const mfem::Vector&);
  typedef double keff_function();
  mfem_function* fun;
  keff_function* analytic_keff;

};

#endif
