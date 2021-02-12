#ifndef VECTOR_INTEGRATOR_WRAPPER
#define VECTOR_INTEGRATOR_WRAPPER

#include "mfem.hpp"

#include <vector>

class VectorIntegratorWrapper : public mfem::BilinearFormIntegrator
{

public:

  VectorIntegratorWrapper(const std::vector<mfem::BilinearFormIntegrator*>& _bfi,
                         int _vdim,
                         const mfem::Array<int>& row_assign,
                         const mfem::Array<int>& col_assign, bool _print=false)
    : bfi{_bfi}, size{static_cast<int>(_bfi.size())}, vdim{_vdim}, 
    row_idx{row_assign}, col_idx{col_assign}, print{_print}
  {
    MFEM_VERIFY((size == row_idx.Size()) && (size == col_idx.Size()), 
      "invalid size of row or column index array");
  }

  virtual void AssembleElementMatrix(const mfem::FiniteElement& el, 
                                     mfem::ElementTransformation& Trans, 
                                     mfem::DenseMatrix& elmat) override;

  virtual void AssembleElementMatrix2(const mfem::FiniteElement& trial_fe, 
                                      const mfem::FiniteElement& test_fe,
                                      mfem::ElementTransformation& Trans, 
                                      mfem::DenseMatrix& elmat) override;

  virtual void AssembleFaceMatrix(const mfem::FiniteElement& el1,
                                  const mfem::FiniteElement& el2,
                                  mfem::FaceElementTransformations& Trans,
                                  mfem::DenseMatrix& elmat) override;

private:

  const std::vector<mfem::BilinearFormIntegrator*>& bfi;
  const int size;
  const int vdim;
  const mfem::Array<int>& row_idx;
  const mfem::Array<int>& col_idx;

  const bool print;

  mfem::DenseMatrix pelmat;

};

#endif
