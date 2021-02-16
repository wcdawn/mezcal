#ifndef VECTOR_INTEGRATOR_WRAPPER
#define VECTOR_INTEGRATOR_WRAPPER

#include "mfem.hpp"

class VectorIntegratorWrapper : public mfem::BilinearFormIntegrator
{

public:

  VectorIntegratorWrapper(const mfem::Array<mfem::BilinearFormIntegrator*>& _bfi,
                         int _vdim,
                         const mfem::Array<int>& row_assign,
                         const mfem::Array<int>& col_assign,
                         const mfem::Array<bool>& own_assign=mfem::Array<bool>())
    : vdim{_vdim}, bfi{_bfi}, row_idx{row_assign}, col_idx{col_assign}
  {
    MFEM_VERIFY((bfi.Size() == row_idx.Size()) && (bfi.Size() == col_idx.Size()), 
      "invalid size of row or column index array");

    // assume ownership of all by default
    if (own_idx.Size() == 0)
    {
      own_idx.SetSize(bfi.Size());
      for (int i{0}; i < own_idx.Size(); ++i)
        own_idx[i] = true;
    }

  }

  VectorIntegratorWrapper(int _vdim, int reserve=0) : vdim{_vdim}
  {
    if (reserve > 0)
    {
      bfi.Reserve(reserve);
      row_idx.Reserve(reserve);
      col_idx.Reserve(reserve);
      own_idx.Reserve(reserve);
    }
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

  void AddIntegrator(mfem::BilinearFormIntegrator* _bfi, int diag, bool own=true);
  void AddIntegrator(mfem::BilinearFormIntegrator* _bfi, int row, int col, bool own=true);

  ~VectorIntegratorWrapper()
  {
    for (int i{0}; i < bfi.Size(); ++i)
      if (own_idx[i])
        delete bfi[i];
  }

private:

  const int vdim;

  mfem::Array<mfem::BilinearFormIntegrator*> bfi;
  mfem::Array<int> row_idx;
  mfem::Array<int> col_idx;
  mfem::Array<bool> own_idx;

  mfem::DenseMatrix pelmat;

};

#endif
