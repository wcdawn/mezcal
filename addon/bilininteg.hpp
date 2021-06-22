// vim: set ts=3 sw=3 sts=3 expandtab:

#ifndef MFEM_BILINTEG_ADDON
#define MFEM_BILINTEG_ADDON

#include "tensor.hpp"

#include "mfem.hpp"

namespace mfem
{

class VectorConvectionIntegrator : public BilinearFormIntegrator
{
protected:
   VectorTensorCoefficient *VTQ{nullptr};
   VectorCoefficient *VQ{nullptr};
   const int vdim;
   const double alpha;
private:
   DenseMatrix pelmat, dshape, adjJ, Q_ir;
   Vector shape, vec1, vec2, BdFidxT;
public:
   VectorConvectionIntegrator(VectorTensorCoefficient& vtq, double a=1.0)
      : VTQ{&vtq}, vdim{vtq.GetVDim()}, alpha{a}
   {}
   VectorConvectionIntegrator(VectorCoefficient& vq, double a=1.0)
      : VQ{&vq}, vdim{vq.GetVDim()}, alpha{a}
   {}
   virtual void AssembleElementMatrix(const FiniteElement &,
                                      ElementTransformation &,
                                      DenseMatrix &) override;
};

class VectorDGTraceIntegrator : public BilinearFormIntegrator
{
protected:
   VectorTensorCoefficient *VTQ{nullptr};
   VectorCoefficient *VQ{nullptr};
   const int vdim;
   const double alpha, beta;
private:
   DenseMatrix pelmat;
   Vector shape1, shape2;
public:
   VectorDGTraceIntegrator(VectorTensorCoefficient& vtq, double a)
      : VTQ{&vtq}, vdim{vtq.GetVDim()}, alpha{a}, beta{0.5*alpha}
   {}
   VectorDGTraceIntegrator(VectorTensorCoefficient& vtq, double a, double b)
      : VTQ{&vtq}, vdim{vtq.GetVDim()}, alpha{a}, beta{b}
   {}
   VectorDGTraceIntegrator(VectorCoefficient &vq, double a)
      : VQ{&vq}, vdim{vq.GetVDim()}, alpha{a}, beta{0.5*alpha}
   {}
   VectorDGTraceIntegrator(VectorCoefficient &vq, double a, double b)
      : VQ{&vq}, vdim{vq.GetVDim()}, alpha{a}, beta{b}
   {}
   using BilinearFormIntegrator::AssembleFaceMatrix;
   virtual void AssembleFaceMatrix(const FiniteElement& el1,
                                   const FiniteElement& el2,
                                   FaceElementTransformations& Trans,
                                   DenseMatrix& elmat) override;
};

class TensorDiffusionIntegrator : public BilinearFormIntegrator
{
protected:
  MatrixTensorCoefficient *MTQ = NULL;
private:
  int vdim{-1};
  DenseMatrix dshape, dshapedxt, pelmat;
public:
  TensorDiffusionIntegrator(MatrixTensorCoefficient& mtq)
    : MTQ{&mtq}, vdim{mtq.GetVDim()} {}
   virtual void AssembleElementMatrix(const FiniteElement &,
                                      ElementTransformation &,
                                      DenseMatrix &) override;
};

} // namespace mfem

#endif
