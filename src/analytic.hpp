#ifndef ANALYTIC
#define ANALYTIC

#include "mfem.hpp"
#include <functional>

using namespace mfem;

class Analytic
{
  public:

    class InvalidReferenceName{};

    enum Reference
    {
      fixed_src,
      oned_oneg,
      twod_oneg,
      square,
      finite_cyl,
      tworeg
    };

    Analytic(Reference ref, const Mesh* const mesh, 
             const GridFunction& grid_sol, double k);

    void analyze(std::ostream& os) const;
    void print(std::ostream& os) const;

  private:
    Reference reference_solution;
    int nVert;
    int nDim;
    double keff;

    // Zeroth zero of the first Bessel function of the first kind.
    static constexpr double alpha0{2.404825557695772768622};

    std::vector<Vertex> node;
    Vector sol;

    std::function<double(const Analytic*, const Vertex&)> fun;
    std::function<double(const Analytic*)> analytic_keff;

    double fixed_src_fun(const Vertex&) const;
    double oned_oneg_fun(const Vertex&) const;
    double twod_oneg_fun(const Vertex&) const;
    double finite_cyl_fun(const Vertex&) const;
    double tworeg_fun(const Vertex&) const;

    double oned_oneg_keff() const;
    double twod_oneg_keff() const;
    double finite_cyl_keff() const;
    double tworeg_keff() const;
};

#endif
