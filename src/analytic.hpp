#ifndef ANALYTIC
#define ANALYTIC

#include "mfem.hpp"
#include "result.hpp"
#include "xslibrary.hpp"

#include <functional>

using namespace mfem;

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

  Analytic(Reference ref, const Result& res, const XSLibrary& xs,
           const Mesh* const mesh);

  void analyze(std::ostream& os) const;
  void print(std::ostream& os) const;

  // helper functions for dealing with Reference enum
  static Reference reference_str_to_enum(std::string str);
  static std::string reference_enum_to_str(Reference ref);

private:
  Reference reference_solution;
  int nVert;
  int nDim;
  double keff;
  const XSLibrary* xs;

  // Zeroth zero of the first Bessel function of the first kind.
  static constexpr double alpha0{2.404825557695772768622};

  std::vector<Vertex> node;
  Vector sol;

  std::function<double(const Analytic*, const Vertex&, const XSLibrary&)> fun;
  std::function<double(const Analytic*, const XSLibrary&)> analytic_keff;

  double fixed_src_fun(const Vertex&, const XSLibrary&) const;
  double oned_oneg_fun(const Vertex&, const XSLibrary&) const;
  double twod_oneg_fun(const Vertex&, const XSLibrary&) const;
  double threed_oneg_fun(const Vertex&, const XSLibrary&) const;
  double finite_cyl_fun(const Vertex&, const XSLibrary&) const;
  double tworeg_fun(const Vertex&, const XSLibrary&) const;

  double oned_oneg_keff(const XSLibrary& xs) const;
  double twod_oneg_keff(const XSLibrary& xs) const;
  double threed_oneg_keff(const XSLibrary& xs) const;
  double finite_cyl_keff(const XSLibrary& xs) const;
  double tworeg_keff(const XSLibrary& xs) const;
};

#endif
