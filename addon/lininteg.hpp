// vim: set ts=3 sw=3 sts=3 expandtab:

#ifndef MFEM_LININTEG_ADDON
#define MFEM_LININTEG_ADDON

#include "tensor.hpp"

#include "mfem.hpp"

namespace mfem
{

#if 0
class VectorDomainLFGradIntegrator : public LinearFormIntegrator
{
private:
  Vector shape;
  VectorTensorCoefficient *VTQ{nullptr};
  VectorCoefficient *VQ{nullptr};
  DenseMatrix dshape;
  const int vdim;
public:
  VectorDomainLFGradIntegrator(VectorTensorCoefficient& vtq)
    : VTQ(&vtq), vdim{vtq.GetVDim()}
  {}
  VectorDomainLFGradIntegrator(VectorCoefficient& vq)
     : VQ(&vq), vdim(vq.GetVDim())
  {}
   virtual void AssembleRHSElementVect(const FiniteElement &el,
                                       ElementTransformation &Tr,
                                       Vector &elvect) override;
   using LinearFormIntegrator::AssembleRHSElementVect;
};
#endif

class VectorBoundaryFlowIntegrator : public LinearFormIntegrator
{
private:
  VectorCoefficient* f;
  VectorTensorCoefficient* u;
  const double alpha, beta;
  const int vdim;
  Vector shape;
public:
  VectorBoundaryFlowIntegrator(VectorCoefficient &_f, 
                               VectorTensorCoefficient &_u,
                               double a)
    : f{&_f}, u{&_u}, alpha{a}, beta{0.5*a}, vdim{_u.GetVDim()}
  {}
  VectorBoundaryFlowIntegrator(VectorCoefficient &_f, 
                               VectorTensorCoefficient &_u, 
                               double a,
                               double b)
    : f{&_f}, u{&_u}, alpha{a}, beta{b}, vdim{_u.GetVDim()}
  {}
   virtual void AssembleRHSElementVect(const FiniteElement &el,
                                       ElementTransformation &Tr,
                                       Vector &elvect) override;
   virtual void AssembleRHSElementVect(const FiniteElement &el,
                                       FaceElementTransformations &Tr,
                                       Vector &elvect) override;
};

} // namespace mfem

#endif
