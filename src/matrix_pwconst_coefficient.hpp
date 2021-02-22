#ifndef MATRIX_PWCONST_COEFFICIENT
#define MATRIX_PWCONST_COEFFICIENT

#include "mfem.hpp"

#include <vector>

class MatrixPWConstCoefficient final : public mfem::MatrixCoefficient
{

public:

  MatrixPWConstCoefficient(int dim, int num_of_sub_divisions, 
                           bool sym=false) 
    : MatrixCoefficient(dim, sym)
  {
    coefficients.resize(num_of_sub_divisions);
  }

  // all DenseMatrix should have the same size but I don't check for that
  // anywhere
  MatrixPWConstCoefficient(const std::vector<mfem::DenseMatrix>& coeff, 
                           bool sym=false) 
    : MatrixCoefficient(coeff[0].Height(), coeff[0].Width(), sym), coefficients{coeff}  // copy construction
  {
  }

  void UpdateMatrices(const std::vector<mfem::DenseMatrix>& coeff) 
  {
    coefficients = coeff; // copy construction
  }

  mfem::DenseMatrix& operator()(int i){return coefficients[i];}
  const mfem::DenseMatrix& operator()(int i) const {return coefficients[i];}

  void operator=(const mfem::DenseMatrix& coeff)
  {
    for (mfem::DenseMatrix& d : coefficients)
      d = coeff;
  }

  void Eval(mfem::DenseMatrix& K, mfem::ElementTransformation& T, 
            const mfem::IntegrationPoint& ip) override;

  int GetNConst() { return coefficients.size(); }

  virtual ~MatrixPWConstCoefficient(){}

private:
  std::vector<mfem::DenseMatrix> coefficients;

};

#endif
