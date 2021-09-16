#ifndef RESULT
#define RESULT

#include "mfem.hpp"

#include <memory>
#include "make_unique.hpp"

struct Result
{

public:

    Result(mfem::ParFiniteElementSpace* space, int ngroup, int nmoment=1);

    // allow indexing with single integer for the scalar flux
    mfem::ParGridFunction& operator()(int g, int n=0) { return *phi[g + n*G]; }
    // const overload
    const mfem::ParGridFunction& operator()(int g, int n=0) const { return *phi[g + n*G]; }

    void operator=(double x)
    {
      for (auto&& flux : phi)
        *flux = x;
    }

    operator std::vector<std::unique_ptr<mfem::ParGridFunction>>&() { return phi; } // implicit converison
    operator const std::vector<std::unique_ptr<mfem::ParGridFunction>>&() const { return phi; } // implicit converison

    double keff{1.0};

    void resize(int ngroup, int nmoment=1);

    int ngroup() const { return G; }
    int nmoment() const { return N; }
    const mfem::ParFiniteElementSpace* FESpace() const { return fespace; }
    mfem::ParFiniteElementSpace* FESpace() { return fespace; }

private:

    mfem::ParFiniteElementSpace* fespace;
    int G; // number of energy groups
    int N; // number of angles/moments
    std::vector<std::unique_ptr<mfem::ParGridFunction>> phi;

    void initialize();

};

#endif
