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

    std::vector<Vertex> node;
    Vector sol;

    std::function<double(const Analytic*, const Vertex&)> fun;
    std::function<double(const Analytic*)> analytic_keff;

    double fixed_src_fun(const Vertex&) const;
    double oned_oneg_fun(const Vertex&) const;

    double oned_oneg_keff() const;
};

#endif
