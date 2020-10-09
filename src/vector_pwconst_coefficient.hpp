#ifndef VECTOR_PWCONST_COEFFICIENT
#define VECTOR_PWCONST_COEFFICIENT

#include "mfem.hpp"

#include <vector>

class VectorPWConstCoefficient final : public mfem::VectorCoefficient
{
public:
  VectorPWConstCoefficient(int vd, int num_of_subdivisions)
    : VectorCoefficient(vd)
  {
    coefficients.resize(num_of_subdivisions);
  }

  VectorPWConstCoefficient(const std::vector<mfem::Vector>& coeff) : VectorCoefficient(coeff[0].Size()), coefficients{coeff} {}

  void UpdateVectors(const std::vector<mfem::Vector>& coeff) {coefficients = coeff;}

  mfem::Vector& operator()(int i){return coefficients[i];}

  void operator=(const mfem::Vector& coeff)
  {
    for (mfem::Vector& v : coefficients)
      v = coeff;
  }

  void Eval(mfem::Vector& V, mfem::ElementTransformation& T, 
            const mfem::IntegrationPoint& ip) override;

  virtual ~VectorPWConstCoefficient(){}

private:
  std::vector<mfem::Vector> coefficients;

};

#endif
