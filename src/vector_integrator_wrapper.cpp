#include "vector_integrator_wrapper.hpp"

using namespace mfem;

void VectorIntegratorWrapper::AssembleElementMatrix(const FiniteElement& el,
                                                    ElementTransformation& Trans,
                                                    DenseMatrix& elmat)
{

  bfi[0]->AssembleElementMatrix(el, Trans, pelmat);
  const int dof{pelmat.Width()};

  elmat.SetSize(vdim*dof);
  elmat = 0.0;
  elmat.AddMatrix(pelmat, row_idx[0], col_idx[0]);

  for (int i{1}; i < bfi.Size(); ++i)
  {
    bfi[i]->AssembleElementMatrix(el, Trans, pelmat);
    elmat.AddMatrix(pelmat, row_idx[i]*dof, col_idx[i]*dof);
  }

  return;

}

void VectorIntegratorWrapper::AssembleElementMatrix2(const FiniteElement& trial_fe,
                                                     const FiniteElement& test_fe,
                                                     ElementTransformation& Trans,
                                                     DenseMatrix& elmat)
{

  bfi[0]->AssembleElementMatrix2(trial_fe, test_fe, Trans, pelmat);
  const int dof{pelmat.Width()};

  elmat.SetSize(vdim*dof);
  elmat = 0.0;
  elmat.AddMatrix(pelmat, row_idx[0], col_idx[0]);

  for (int i{1}; i < bfi.Size(); ++i)
  {
    bfi[i]->AssembleElementMatrix2(trial_fe, test_fe, Trans, pelmat);
    elmat.AddMatrix(pelmat, row_idx[i]*dof, col_idx[i]*dof);
  }

  return;

}

void VectorIntegratorWrapper::AssembleFaceMatrix(const FiniteElement& el1,
                                                 const FiniteElement& el2,
                                                 FaceElementTransformations& Trans,
                                                 DenseMatrix& elmat)
{

  bfi[0]->AssembleFaceMatrix(el1, el2, Trans, pelmat);
  const int dof{pelmat.Width()};

  elmat.SetSize(vdim*dof);
  elmat = 0.0;
  elmat.AddMatrix(pelmat, row_idx[0], col_idx[0]);

  for (int i{1}; i < bfi.Size(); ++i)
  {
    bfi[i]->AssembleFaceMatrix(el1, el2, Trans, pelmat);
    elmat.AddMatrix(pelmat, row_idx[i]*dof, col_idx[i]*dof);
  }

  return;

}

void VectorIntegratorWrapper::AddIntegrator(BilinearFormIntegrator* _bfi, 
                                            int diag, bool own)
{
  AddIntegrator(_bfi, diag, diag, own);
}

void VectorIntegratorWrapper::AddIntegrator(BilinearFormIntegrator* _bfi, 
                                            int row, int col, bool own)
{
  bfi.Append(_bfi);
  row_idx.Append(row);
  col_idx.Append(col);
  own_idx.Append(own);
}
