#ifndef TENSOR_CLASS
#define TENSOR_CLASS

#include <vector>
#include <utility>

#include "mfem.hpp"

namespace mfem
{

template<typename T>
class Tensor
{

public:

  Tensor() {}

  void AddElement(const T& elem, int row, int col)
  {
    data.emplace_back(elem); // copy construction
    row_idx.Append(row);
    col_idx.Append(col);
  }

  void AddElement(const T& elem, int dia)
  {
    AddElement(elem, dia, dia);
  }

  void Eval(std::vector<T>& data_out, 
            Array<int>& row_idx_out,
            Array<int>& col_idx_out) const
  {
    data_out    = data; // copy
    row_idx_out = row_idx;
    col_idx_out = col_idx;
  }

  void reserve(int res)
  {
    data.reserve(res);
    row_idx.Reserve(res);
    col_idx.Reserve(res);
  }

  void Transpose()
  {
    std::swap(row_idx, col_idx);
  }

private:

  std::vector<T> data;
  Array<int> row_idx;
  Array<int> col_idx;

};

class MatrixTensorCoefficient
{

public:

  MatrixTensorCoefficient(int _vdim) : vdim{_vdim} {}

  virtual void Eval(std::vector<DenseMatrix>& data,
                    Array<int>& row_idx,
                    Array<int>& col_idx,
                    ElementTransformation& T,
                    const IntegrationPoint& ip) const = 0;

  int GetVDim() const { return vdim; }

  virtual ~MatrixTensorCoefficient() {}

protected:

  const int vdim;

};

class VectorTensorCoefficient
{

public:

  VectorTensorCoefficient(int _vdim) : vdim{_vdim} {}

  virtual void Eval(std::vector<Vector>& data,
                    Array<int>& row_idx,
                    Array<int>& col_idx,
                    ElementTransformation& T,
                    const IntegrationPoint& ip) const = 0;

  int GetVDim() const { return vdim; }

  virtual ~VectorTensorCoefficient() {}

protected:

  const int vdim;

};

class MatrixTensorPWConstCoefficient : public MatrixTensorCoefficient
{

public:

  MatrixTensorPWConstCoefficient(int _vdim, 
                                 const std::vector<Tensor<DenseMatrix>>& coeff)
    : MatrixTensorCoefficient{_vdim}, coefficients{coeff}
  {}

  virtual void Eval(std::vector<DenseMatrix>& data,
                    Array<int>& row_idx,
                    Array<int>& col_idx,
                    ElementTransformation& T,
                    const IntegrationPoint& ip) const override
  {
    const int att{T.Attribute-1};
    coefficients[att].Eval(data, row_idx, col_idx);
  }

private:

  const std::vector<Tensor<DenseMatrix>>& coefficients;

};

class MatrixTensorConstantCoefficient : public MatrixTensorCoefficient
{

public:

  MatrixTensorConstantCoefficient(int _vdim, 
                                  const Tensor<DenseMatrix>& coeff)
    : MatrixTensorCoefficient{_vdim}, coefficient{coeff}
  {}

  virtual void Eval(std::vector<DenseMatrix>& data,
                    Array<int>& row_idx,
                    Array<int>& col_idx,
                    ElementTransformation& T,
                    const IntegrationPoint& ip) const override
  {
    coefficient.Eval(data, row_idx, col_idx);
  }

private:

  const Tensor<DenseMatrix>& coefficient;

};

class VectorTensorPWConstCoefficient : public VectorTensorCoefficient
{

public:

  VectorTensorPWConstCoefficient(int _vdim, 
                                 const std::vector<Tensor<Vector>>& coeff)
    : VectorTensorCoefficient{_vdim}, coefficients{coeff}
  {}

  virtual void Eval(std::vector<Vector>& data,
                    Array<int>& row_idx,
                    Array<int>& col_idx,
                    ElementTransformation& T,
                    const IntegrationPoint& ip) const override
  {
    const int att{T.Attribute-1};
    coefficients[att].Eval(data, row_idx, col_idx);
  }

private:

  std::vector<Tensor<Vector>> coefficients;

};

class VectorTensorConstantCoefficient : public VectorTensorCoefficient
{

public:

  VectorTensorConstantCoefficient(int _vdim,
                                  const Tensor<Vector>& coeff)
    : VectorTensorCoefficient{_vdim}, coefficient{coeff}
  {}


  virtual void Eval(std::vector<Vector>& data,
                    Array<int>& row_idx,
                    Array<int>& col_idx,
                    ElementTransformation& T,
                    const IntegrationPoint& ip) const override
  {
    coefficient.Eval(data, row_idx, col_idx);
  }

private:

  const Tensor<Vector> coefficient; // copy TODO


};  

class ScalarVectorTensorProductCoefficient : public VectorTensorCoefficient
{
public :
  ScalarVectorTensorProductCoefficient(Coefficient& q, VectorTensorCoefficient& vtq)
    : VectorTensorCoefficient{vtq.GetVDim()}, Q{&q}, VTQ{&vtq}
  {}

  virtual void Eval(std::vector<Vector>& data,
                    Array<int>& row_idx,
                    Array<int>& col_idx,
                    ElementTransformation& T,
                    const IntegrationPoint& ip) const override
  {
    VTQ->Eval(data, row_idx, col_idx, T, ip);
    if (Q)
    {
      const double val{Q->Eval(T, ip)};
      for (Vector& v : data)
        v *= val;
    }
  }

private:
  Coefficient* Q{nullptr};
  VectorTensorCoefficient* VTQ{nullptr};
};

class ScalarMatrixTensorProductCoefficient : public MatrixTensorCoefficient
{
public :
  ScalarMatrixTensorProductCoefficient(Coefficient& q, MatrixTensorCoefficient& vtq)
    : MatrixTensorCoefficient{vtq.GetVDim()}, Q{&q}, VTQ{&vtq}
  {}

  virtual void Eval(std::vector<DenseMatrix>& data,
                    Array<int>& row_idx,
                    Array<int>& col_idx,
                    ElementTransformation& T,
                    const IntegrationPoint& ip) const override
  {
    VTQ->Eval(data, row_idx, col_idx, T, ip);
    if (Q)
    {
      const double val{Q->Eval(T, ip)};
      for (DenseMatrix& v : data)
        v *= val;
    }
  }

private:
  Coefficient* Q{nullptr};
  MatrixTensorCoefficient* VTQ{nullptr};
};

} // namespace mfem

#endif
