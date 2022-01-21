// vim: set ts=3 sw=3 sts=3 expandtab:

#include <algorithm>

#include "bilininteg.hpp"

namespace mfem
{

void VectorConvectionIntegrator::AssembleElementMatrix(
   const FiniteElement &el, ElementTransformation &Trans, DenseMatrix &elmat)
{
   const int nd = el.GetDof();
   const int dim = el.GetDim();
   const int spaceDim = Trans.GetSpaceDim();

   elmat.SetSize(nd*vdim);
   pelmat.SetSize(nd);
   dshape.SetSize(nd,dim);
   adjJ.SetSize(dim);
   shape.SetSize(nd);
   vec2.SetSize(dim);
   BdFidxT.SetSize(nd);

   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
      int order = Trans.OrderGrad(&el) + Trans.Order() + el.GetOrder();
      ir = &IntRules.Get(el.GetGeomType(), order);
   }

   elmat = 0.0;
   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);
      el.CalcDShape(ip, dshape);
      el.CalcShape(ip, shape);

      Trans.SetIntPoint(&ip);
      CalcAdjugate(Trans.Jacobian(), adjJ);

      if (VTQ)
      {
         std::vector<Vector> vec_data;
         mfem::Array<int> row_idx, col_idx;
         VTQ->Eval(vec_data, row_idx, col_idx, Trans, ip);
         for (int idx = 0; idx < row_idx.Size(); ++idx)
         {
            vec_data[idx] *= alpha * ip.weight;
            adjJ.Mult(vec_data[idx], vec2);
            dshape.Mult(vec2, BdFidxT);
            pelmat = 0.0;
            AddMultVWt(shape, BdFidxT, pelmat);
            elmat.AddMatrix(pelmat, row_idx[idx]*nd, col_idx[idx]*nd);
         }
      }
      else if (VQ)
      {
         Vector x(vdim);
         VQ->Eval(x, Trans, ip);
         for (int idx{0}; idx < vdim/spaceDim; ++idx)
         {
            Vector part_x(spaceDim);
            for (int jdx{0}; jdx < spaceDim; ++jdx)
               part_x(jdx) = x(jdx+idx*spaceDim);
            part_x *= alpha * ip.weight;
            adjJ.Mult(part_x, vec2);
            dshape.Mult(vec2, BdFidxT);
            pelmat = 0.0;
            AddMultVWt(shape, BdFidxT, pelmat);
            elmat.AddMatrix(pelmat, idx*nd, idx*nd);
         }
      }
      else
      {
         MFEM_ABORT("undefined coefficient for some reason");
      }

   }
}

void VectorDGTraceIntegrator::AssembleFaceMatrix(const FiniteElement& el1,
                                                 const FiniteElement& el2,
                                                 FaceElementTransformations& Trans,
                                                 DenseMatrix& elmat)
{
   int dim, ndof1, ndof2;

   double un, a, b, w;

   dim = el1.GetDim();
   ndof1 = el1.GetDof();
   Vector nor(dim);

   const int spaceDim{Trans.GetSpaceDim()};

   if (Trans.Elem2No >= 0)
   {
      ndof2 = el2.GetDof();
   }
   else
   {
      ndof2 = 0;
   }

   shape1.SetSize(ndof1);
   shape2.SetSize(ndof2);
   elmat.SetSize((ndof1 + ndof2) * vdim);
   pelmat.SetSize(ndof1 + ndof2);
   elmat = 0.0;

   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
      int order;
      // Assuming order(u)==order(mesh)
      if (Trans.Elem2No >= 0)
         order = (std::min(Trans.Elem1->OrderW(), Trans.Elem2->OrderW()) +
                  2*std::max(el1.GetOrder(), el2.GetOrder()));
      else
      {
         order = Trans.Elem1->OrderW() + 2*el1.GetOrder();
      }
      if (el1.Space() == FunctionSpace::Pk)
      {
         order++;
      }
      ir = &IntRules.Get(Trans.GetGeometryType(), order);
   }

   for (int p = 0; p < ir->GetNPoints(); p++)
   {
      const IntegrationPoint &ip = ir->IntPoint(p);

      // Set the integration point in the face and the neighboring elements
      Trans.SetAllIntPoints(&ip);

      // Access the neighboring elements' integration points
      // Note: eip2 will only contain valid data if Elem2 exists
      const IntegrationPoint &eip1 = Trans.GetElement1IntPoint();
      const IntegrationPoint &eip2 = Trans.GetElement2IntPoint();

      el1.CalcShape(eip1, shape1);

      if (dim == 1)
      {
         nor(0) = 2*eip1.x - 1.0;
      }
      else
      {
         CalcOrtho(Trans.Jacobian(), nor);
      }

      if (VTQ)
      {

         std::vector<Vector> vu;
         Array<int> row_idx, col_idx;
         VTQ->Eval(vu, row_idx, col_idx, *Trans.Elem1, eip1);
         for (int idx = 0; idx < row_idx.Size(); ++idx)
         {
            un = vu[idx] * nor;
            a = 0.5 * alpha * un;
            b = beta * fabs(un);

            // note: if |alpha/2|==|beta| then |a|==|b|, i.e. (a==b) or (a==-b)
            //       and therefore two blocks in the element matrix contribution
            //       (from the current quadrature point) are 0
            w = ip.weight * (a+b);
            if (w != 0.0)
            {
               for (int i = 0; i < ndof1; i++)
               {
                  const int row{i+row_idx[idx]*(ndof1+ndof2)};
                  for (int j = 0; j < ndof1; j++)
                  {
                     const int col{j+col_idx[idx]*(ndof1+ndof2)};
                     elmat(row, col) += w * shape1(i) * shape1(j);
                  }
               }
            }

            if (ndof2)
            {
               el2.CalcShape(eip2, shape2);

               if (w != 0.0)
               {
                  for (int i = 0; i < ndof2; i++)
                  {
                     const int row{ndof1+i+row_idx[idx]*(ndof1+ndof2)};
                     for (int j = 0; j < ndof1; j++)
                     {
                        const int col{j+col_idx[idx]*(ndof1+ndof2)};
                        elmat(row, col) -= w * shape2(i) * shape1(j);
                     }
                  }
               }

               w = ip.weight * (b-a);
               if (w != 0.0)
               {
                  for (int i = 0; i < ndof2; i++)
                  {
                     for (int j = 0; j < ndof2; j++)
                     {
                        elmat(ndof1+i, ndof1+j) += w * shape2(i) * shape2(j);
                     }
                  }

                  for (int i = 0; i < ndof1; i++)
                  {
                     for (int j = 0; j < ndof2; j++)
                     {
                        elmat(i, ndof1+j) -= w * shape1(i) * shape2(j);
                     }
                  }
               }
            } // ndof2
         } // VTQ

      }
      else if (VQ)
      {

         Vector x(vdim);
         VQ->Eval(x, *Trans.Elem1, eip1);
         for (int idx = 0; idx < vdim/spaceDim; ++idx)
         {
            Vector part_x(spaceDim);
            for (int jdx{0}; jdx < spaceDim; ++jdx)
               part_x(jdx) = x(jdx+idx*spaceDim);
            un = part_x * nor;
            a = 0.5 * alpha * un;
            b = beta * fabs(un);

            // note: if |alpha/2|==|beta| then |a|==|b|, i.e. (a==b) or (a==-b)
            //       and therefore two blocks in the element matrix contribution
            //       (from the current quadrature point) are 0
            w = ip.weight * (a+b);
            if (w != 0.0)
            {
               for (int i = 0; i < ndof1; i++)
               {
                  const int row{i+idx*(ndof1+ndof2)};
                  for (int j = 0; j < ndof1; j++)
                  {
                     const int col{j+idx*(ndof1+ndof2)};
                     elmat(row, col) += w * shape1(i) * shape1(j);
                  }
               }
            }

            if (ndof2)
            {
               el2.CalcShape(eip2, shape2);

               if (w != 0.0)
               {
                  for (int i = 0; i < ndof2; i++)
                  {
                     const int row{ndof1+i+idx*(ndof1+ndof2)};
                     for (int j = 0; j < ndof1; j++)
                     {
                        const int col{j+idx*(ndof1+ndof2)};
                        elmat(row, col) -= w * shape2(i) * shape1(j);
                     }
                  }
               }

               w = ip.weight * (b-a);
               if (w != 0.0)
               {
                  for (int i = 0; i < ndof2; i++)
                  {
                     for (int j = 0; j < ndof2; j++)
                     {
                        elmat(ndof1+i, ndof1+j) += w * shape2(i) * shape2(j);
                     }
                  }

                  for (int i = 0; i < ndof1; i++)
                  {
                     for (int j = 0; j < ndof2; j++)
                     {
                        elmat(i, ndof1+j) -= w * shape1(i) * shape2(j);
                     }
                  }
               }
            } // ndof2
         }

      }
      else
      {
         MFEM_ABORT("undefined coefficient for some reason");
      }


   }
}

void TensorDiffusionIntegrator::AssembleElementMatrix(
   const FiniteElement &el,
   ElementTransformation &Trans,
   DenseMatrix &elmat)
{
   const int dim = el.GetDim();
   const int dof = el.GetDof();
   const int sdim = Trans.GetSpaceDim();

   // If vdim is not set, set it to the space dimension;
   vdim = (vdim <= 0) ? sdim : vdim;
   const bool square = (dim == sdim);

   dshape.SetSize(dof, dim);
   dshapedxt.SetSize(dof, sdim);
   dshapedxt_m.SetSize(dof, MTQ ? sdim : 0);

   elmat.SetSize(vdim * dof);
   pelmat.SetSize(dof);

   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
      ir = &DiffusionIntegrator::GetRule(el,el);
   }

   elmat = 0.0;

   for (int i = 0; i < ir -> GetNPoints(); i++)
   {

      const IntegrationPoint &ip = ir->IntPoint(i);
      el.CalcDShape(ip, dshape);

      Trans.SetIntPoint(&ip);
      double w = Trans.Weight();
      w = ip.weight / (square ? w : w*w*w);
      // AdjugateJacobian = / adj(J),         if J is square
      //                    \ adj(J^t.J).J^t, otherwise
      Mult(dshape, Trans.AdjugateJacobian(), dshapedxt);
      if (MTQ)
      {
         std::vector<DenseMatrix> mat_data;
         Array<int> row_idx, col_idx;
         MTQ->Eval(mat_data, row_idx, col_idx, Trans, ip);
         for (int idx{0}; idx < row_idx.Size(); ++idx)
         {
            pelmat = 0.0;
            mat_data[idx] *= w;
            Mult(dshapedxt, mat_data[idx], dshapedxt_m);
            MultABt(dshapedxt_m, dshapedxt, pelmat);
            elmat.AddMatrix(pelmat, row_idx[idx]*dof, col_idx[idx]*dof);
         }
      }
      else
      {
         Mult_a_AAt(w, dshapedxt, pelmat);
         for (int k = 0; k < vdim; ++k)
         {
            elmat.AddMatrix(pelmat, dof*k, dof*k);
         }
      }
   }
}

} // namespace mfem
