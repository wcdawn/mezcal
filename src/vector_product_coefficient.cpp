#include "vector_product_coefficient.hpp"

using namespace mfem;

void VectorProductCoefficient::Eval(Vector &V,
                                    ElementTransformation &T,
                                    const IntegrationPoint &ip)
{

  Vector a_vec(vdim);
  a.Eval(a_vec, T, ip);

  Vector b_vec(vdim);
  b.Eval(b_vec, T, ip);

  V.SetSize(vdim);
  for (int i{0}; i < vdim; ++i)
    V(i) = a_vec(i)*b_vec(i);

  return;
}
