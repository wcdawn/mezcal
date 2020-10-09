#ifndef SPN_SOURCE_COEFFICIENT
#define SPN_SOURCE_COEFFICIENT

#include "material_library.hpp"
#include "mfem.hpp"
#include "xslibrary.hpp"
#include "result.hpp"

#include <vector>

class SPNSourceCoefficient : public mfem::VectorCoefficient
{

public:

  SPNSourceCoefficient(int group, int order, int maximum_order,
                       const Material_library& mat_lib_lib,
                       const Result& flux)
    : VectorCoefficient{flux(0,0).FESpace()->GetMesh()->Dimension()},
    dim{flux(0,0).FESpace()->GetMesh()->Dimension()},
    g{group}, 
    n{order},
    dble_n{2.0*order},
    max_ord{maximum_order},
    mat_lib{mat_lib_lib},
    phi{flux},
    xmult_next{(2.0*order+1.0)*(2.0*order+2.0) / ((2.0*2.0*order+1.0)*(2.0*2.0*order+3.0))},
    xmult_prev{(2.0*order*2.0*order - 2.0*order) / (4.0*2.0*order*2.0*order - 1.0)}
  {
  }

  virtual void Eval(mfem::Vector& V, mfem::ElementTransformation& T, 
                      const mfem::IntegrationPoint& ip) override;

private:

  // Number of spatial dimensions.
  // This will also be the dimension of the Vectors.
  const int dim;

  // Current energy group.
  const int g;

  // Current SPN half-order.
  // This is the same as the index of this SPN equation.
  // i.e. n=0...(N+1)/2 and dble_n=2.0*n
  const int n;
  const double dble_n;

  // Truncation half-order.
  // This is the same as the total number of SPN equations.
  // i.e. max_ord=(N+1)/2
  // e.g. for SP3, max_order=2.
  const int max_ord;

  // Store a reference to the cross section library.
  // We count on the fact that it does not go out of scope while we need it.
  const Material_library& mat_lib;

  // Store a reference to the flux vector.
  const Result& phi;

  const double xmult_next;
  const double xmult_prev;

};

#endif
