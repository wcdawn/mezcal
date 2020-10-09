#ifndef BOUNDARY_ABS_NORMAL_BF_INTEGRATOR
#define BOUNDARY_ABS_NORMAL_BF_INTEGRATOR

#include "mfem.hpp"

namespace mfem
{
  class BoundaryAbsNormalBFIntegrator final : public BilinearFormIntegrator
  {

  public:

    BoundaryAbsNormalBFIntegrator(VectorCoefficient &vq, int a=1, int b=1)
      : VQ{vq}, oa{a}, ob{b} {}

    using BilinearFormIntegrator::AssembleFaceMatrix;

    virtual void AssembleFaceMatrix(const FiniteElement& el1,
                                    const FiniteElement& el2,
                                    FaceElementTransformations& Trans,
                                    DenseMatrix& elmat);

  private:

    Vector shape;
    VectorCoefficient& VQ;
    const int oa, ob;

  };
}

#endif
