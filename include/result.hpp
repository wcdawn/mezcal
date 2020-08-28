#ifndef RESULT
#define RESULT

#include "mfem.hpp"

struct Result
{
  public:
    Result() {}
    Result(mfem::ParFiniteElementSpace* fespace, int ngroup) : G{ngroup}
    { 
      phi.resize(G);
      for (int g{0}; g < G; ++g)
      {
        phi[g].SetSpace(fespace);
        // must be zero to enforce zero-flux (essential) boundary conditions
        phi[g] = 0.0;
      }
    }

    std::vector<mfem::ParGridFunction> phi;
    double keff{1.0};

  private:
    int G{};

};

#endif
