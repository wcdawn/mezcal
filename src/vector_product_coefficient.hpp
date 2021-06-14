#ifndef VECTOR_PRODUCT_COEFFICIENT_HPP
#define VECTOR_PRODUCT_COEFFICIENT_HPP

#include "mfem.hpp"

// performs element-wise multiplication of two vector coefficients
// result_i = a_i * b_i

class VectorProductCoefficient final : public mfem::VectorCoefficient
{

public:

  VectorProductCoefficient(mfem::VectorCoefficient &a_in, mfem::VectorCoefficient &b_in)
    : VectorCoefficient(a_in.GetVDim()), a{a_in}, b{b_in}
  {
  }

  void Eval(mfem::Vector &V,
            mfem::ElementTransformation &T,
            const mfem::IntegrationPoint &ip) override;

  virtual ~VectorProductCoefficient(){}

private:

  mfem::VectorCoefficient &a;
  mfem::VectorCoefficient &b;

};

#endif
