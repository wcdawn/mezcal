#include "result.hpp"

using namespace mfem;
using std::size_t;

Result::Result(ParFiniteElementSpace* space, int ngroup, int nmoment)
  : fespace{space}, G{ngroup}, N{nmoment}
{
  phi.resize(G*N);
  initialize();
}

void Result::resize(int ngroup, int nmoment)
{
  const int old_group{G};
  const int old_nmoment{N};

  G = ngroup;
  N = nmoment;

  if (static_cast<size_t>(G*N) > phi.size())
  {
    phi.resize(G*N);

    // if we introduce new energy groups, zero out all the fluxes because we
    // have no idea what the new vs. old energy structure is
    if (old_group != G)
    {
      initialize();
      return;
    }

    // if we introduce new moments, the old values are already in their correct
    // locations, we just need to zero out the new values
    if (old_nmoment != N)
    {
      for (int g{0}; g < G; ++g)
      {
        for (int n{old_nmoment}; n < N; ++n)
        {
          const int idx{g+G*n};
          phi[idx].SetSpace(fespace);
          phi[idx] = 0.0;
        }
      }
    }

  }

  return;

}

void Result::initialize()
{
  for (ParGridFunction& flux : phi)
  {
    flux.SetSpace(fespace);
    // must be zero to enforce zero-flux (essential) boundary conditions
    flux = 0.0;
  }
}
