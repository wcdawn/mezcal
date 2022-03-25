// vim: set ts=3 sw=3 sts=3 expandtab:

#include "lininteg.hpp"

namespace mfem
{

#if 0
void VectorDomainLFGradIntegrator::AssembleRHSElementVect(
   const FiniteElement &el, ElementTransformation &Tr, Vector &elvect)
{
   int dof = el.GetDof();
   int spaceDim = Tr.GetSpaceDim();

   dshape.SetSize(dof, spaceDim);

   Vector pelvect(dof);
   elvect.SetSize(dof*vdim/spaceDim);
   elvect = 0.0;

   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
      int intorder = 2 * el.GetOrder();
      ir = &IntRules.Get(el.GetGeomType(), intorder);
   }

   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);

      Tr.SetIntPoint(&ip);
      el.CalcPhysDShape(Tr, dshape);

      if (VTQ)
      {
         std::vector<Vector> vec_data;
         Array<int> row_idx, col_idx;
         VTQ->Eval(vec_data, row_idx, col_idx, Tr, ip);
         for (int idx{0}; idx < row_idx.Size(); ++idx)
         {
            MFEM_ASSERT(row_idx[idx] == col_idx[idx], 
               "row_idx & col_idx must be the same for tensors in VectorDomainLFGradIntegrator");
            vec_data[idx] *= ip.weight * Tr.Weight();
            dshape.Mult(vec_data[idx], pelvect);
            for (int j{0}; j < dof; ++j)
            {
               elvect(j+dof*row_idx[idx]) += pelvect(j);
            }
         }
      }
      else if (VQ)
      {
         Vector x(vdim);
         VQ->Eval(x, Tr, ip);
         x *= ip.weight * Tr.Weight();
         for (int idx{0}; idx < vdim/spaceDim; ++idx)
         {
            Vector part_x(spaceDim);
            for (int jdx{0}; jdx < spaceDim; ++jdx)
               part_x(jdx) = x(idx*spaceDim+jdx);
            dshape.Mult(part_x, pelvect);
            for (int j{0}; j < dof; ++j)
            {
               elvect(j+dof*idx) += pelvect(j);
            }
         }
      }
      else
      {
         MFEM_ABORT("undefined coefficient for some reason");
      }
   }
}
#endif

void VectorBoundaryFlowIntegrator::AssembleRHSElementVect(
   const FiniteElement &el, ElementTransformation &Tr, Vector &elvect)
{
   mfem_error("VectorBoundaryFlowIntegrator::AssembleRHSElementVect\n"
              "  is only implemented as boundary integrator!\n"
              "  Use LinearForm::AddBdrFaceIntegrator instead of\n"
              "  LinearForm::AddBoundaryIntegrator.");
}

void VectorBoundaryFlowIntegrator::AssembleRHSElementVect(
   const FiniteElement &el, FaceElementTransformations &Tr, Vector &elvect)
{
   int dim, ndof, order;
   double un, w, nor_data[3];

   dim  = el.GetDim();
   ndof = el.GetDof();
   Vector nor(nor_data, dim);

   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
      // Assuming order(u)==order(mesh)
      order = Tr.Elem1->OrderW() + 2*el.GetOrder();
      if (el.Space() == FunctionSpace::Pk)
      {
         order++;
      }
      ir = &IntRules.Get(Tr.GetGeometryType(), order);
   }

   shape.SetSize(ndof);
   elvect.SetSize(ndof*vdim);
   elvect = 0.0;

   std::vector<Vector> u_data;
   Array<int> row_idx, col_idx;
   Vector f_data(vdim);

   for (int p = 0; p < ir->GetNPoints(); p++)
   {
      const IntegrationPoint &ip = ir->IntPoint(p);

      // Set the integration point in the face and the neighboring element
      Tr.SetAllIntPoints(&ip);

      // Access the neighboring element's integration point
      const IntegrationPoint &eip = Tr.GetElement1IntPoint();
      el.CalcShape(eip, shape);

      if (dim == 1)
      {
         nor(0) = 2*eip.x - 1.0;
      }
      else
      {
         CalcOrtho(Tr.Jacobian(), nor);
      }

      // Use Tr.Elem1 transformation for u so that it matches the coefficient
      // used with the ConvectionIntegrator and/or the DGTraceIntegrator.
      u->Eval(u_data, row_idx, col_idx, *Tr.Elem1, eip);
      //f->Eval(f_data, *Tr.Elem1, eip);
      f->Eval(f_data, Tr, eip);

      for (int idx{0}; idx < row_idx.Size(); ++idx)
      {
        // TODO remove this
        MFEM_ASSERT(row_idx[idx] == col_idx[idx], "VectorTensorCoefficient must be sort-of-diagonal");
        un = u_data[idx] * nor;
        w = 0.5*alpha*un - beta*fabs(un);
        w *= ip.weight*f_data(row_idx[idx]);
        for (int j{0}; j < ndof; ++j)
        {
          elvect(j+row_idx[idx]*ndof) += w*shape(j);
        }
      }

   }
}

} // namespace mfem
