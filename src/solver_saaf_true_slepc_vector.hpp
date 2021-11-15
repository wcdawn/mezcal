#ifndef SOLVER_SAAF_TRUE_SLEPC_VECTOR_HPP
#define SOLVER_SAAF_TRUE_SLEPC_VECTOR_HPP

#include "input.hpp"
#include "material_library.hpp"
#include "quadrature.hpp"

#include "result.hpp"
#include "timer.hpp"

#include "mfem.hpp"
#include "petsc.h"

void saaf_true_slepc_vector(const Quadrature* const quad,
                            Timer& time,
                            const Input& inp,
                            const Material_library& mat_lib,
                            Result& angular_flux);

/*
 * WARNING
 * Code in this anonymous namespace is duplicated in solver.[ch]pp.
 * WARNING
 */
namespace
{

namespace slepc_amgx
{

  extern std::unique_ptr<mfem::Solver> sol;
  extern mfem::HypreParMatrix* Ahypre;

  PetscErrorCode setup(PC pc);
  PetscErrorCode apply(PC pc, Vec xin, Vec xout);
  PetscErrorCode destroy(PC pc);

} // namespace slepc_amgx

namespace slepc_hypre
{
  extern std::unique_ptr<mfem::Solver> sol;
  extern std::unique_ptr<mfem::HypreBoomerAMG> amg;
  extern const mfem::HypreParMatrix *Ahypre;

  extern const Input *inp;
  extern const Material_library *mat_lib;
  extern const Quadrature *quad;

  PetscErrorCode setup(PC pc);
  PetscErrorCode apply(PC pc, Vec xin, Vec xout);
  PetscErrorCode destroy(PC pc);

} // namespace slepc_hypre

mfem::HypreParMatrix* saaf_a_matrix(const Input &inp, 
                                    Timer &time,
                                    const Material_library &mat_lib,
                                    const Quadrature* const quad,
                                    mfem::ParFiniteElementSpace* fespace);
mfem::HypreParMatrix* saaf_m_matrix(const Input &inp, 
                                    Timer &time,
                                    const Material_library &mat_lib,
                                    const Quadrature* const quad,
                                    mfem::ParFiniteElementSpace* fespace);

void slepc_initialize(const Input& inp);
void slepc_solver_wrapper(mfem::HypreParMatrix* A, mfem::HypreParMatrix* M, const Input& inp, Timer& time, Result& res, int numfunctions=1);
void slepc_finalize();

void flux_sign_fixup(mfem::Vector &v);

} // namespace anonymous

#endif
