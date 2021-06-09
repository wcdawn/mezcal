#ifndef SOLVER_HPP
#define SOLVER_HPP

#include "input.hpp"
#include "material_library.hpp"

#include "mfem.hpp"
#include "../addon/mfem_addon.hpp"

#include "result.hpp"
#include "timer.hpp"
#include "xslibrary.hpp"
#include "quadrature.hpp"

#include <exception>
#include <string>
#include <vector>

#include "petsc.h"

namespace slepc_amgx
{

  extern std::unique_ptr<mfem::AmgXSolver> sol;
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

namespace slepc_diffusion
{

  extern std::unique_ptr<mfem::Solver> sol;
  extern std::unique_ptr<mfem::HypreParMatrix> Alo;

  extern std::unique_ptr<mfem::HypreBoomerAMG> amg;

  extern const Input* inp;
  extern const Material_library* mat_lib;
  extern mfem::ParGridFunction *angular_flux;
  extern mfem::ParGridFunction *scalar_flux;

  PetscErrorCode setup(PC pc);
  PetscErrorCode apply(PC pc, Vec xin, Vec xout);
  PetscErrorCode destroy(PC pc);

} // namespace slepc_diffusion

namespace slepc_transport
{

  extern std::unique_ptr<mfem::Solver> sol;
  extern std::unique_ptr<mfem::HypreParMatrix> Alo;

  extern std::unique_ptr<mfem::HypreBoomerAMG> amg;

  extern const Input *inp;
  extern const Material_library *mat_lib;
  extern const Quadrature *q_lo;
  extern const Quadrature *q_hi;
  extern mfem::ParFiniteElementSpace *fespace_lo;

  extern int fine_per_coarse;

  PetscErrorCode setup(PC pc);
  PetscErrorCode apply(PC pc, Vec xin, Vec xout);
  PetscErrorCode destroy(PC pc);

} // namespace slepc_transport

namespace slepc_split
{

  extern std::unique_ptr<mfem::Solver> sol;
  extern std::unique_ptr<mfem::HypreBoomerAMG> amg;
  extern std::unique_ptr<mfem::HypreParMatrix> Alc_hypre;

  extern const Input *inp;
  extern const Material_library *mat_lib;
  extern const Quadrature *quad;

  extern Result *angular_flux;
  extern Result *scalar_flux;

  // TODO I stopped keeping up. There are *a lot* more pointers floating around
  // that either need to be listed here or gotten rid of.
  extern std::unique_ptr<mfem::ParLinearForm> scat_src;

  PetscErrorCode setup(PC pc);
  PetscErrorCode apply(PC pc, Vec xin, Vec xout);
  PetscErrorCode destroy(PC pc);

} // namespace slepc_split

template<typename T>
class SPNWrapper
{

public:

  SPNWrapper(int ngroup, int nmoment=1) : G{ngroup}, N{nmoment} {}

  T& operator()(int g, int n=0) { return arr[g+G*n]; }
  const T& operator()(int g, int n=0) const { return arr[g+G*n]; }

  void reserve() { arr.reserve(G*N); }
  void resize() { arr.resize(G*N); }

  template <class... Args>
  void emplace_back(Args&&... args) { arr.emplace_back(args...); }

  template <class... Args>
  void push_back(Args&&... args) { arr.push_back(args...); }

  int ngroup() const { return G; }
  int nmoment() const { return N; }

  T& back() { return arr.back(); }
  const T& back() const { return arr.back(); }

private:

  const int G;
  const int N;
  std::vector<T> arr;

};

void diffusion_hypre_lobpcg(Timer& time, const Input& inp,
                              const Material_library& mat_lib,
                              const mfem::Array<int>& ess_bdr,
                              const mfem::ParMesh* const pmesh,
                              mfem::ParFiniteElementSpace* const fespace, Result& res);

void spn_power_iteration(Timer& time, const Input& inp,
                         Material_library& mat_lib,
                         mfem::Array<int>& ess_bdr,
                         const mfem::ParMesh* const pmesh,
                         mfem::ParFiniteElementSpace* const fespace, Result& res);
void spn_amgx_power_iteration(Timer& time, const Input& inp,
                              Material_library& mat_lib,
                              mfem::Array<int>& ess_bdr,
                              const mfem::ParMesh* const pmesh,
                              mfem::ParFiniteElementSpace* const fespace, Result& res);

void diffusion_fixed_src(Timer& time, const Input& inp,
                           const Material_library& mat_lib,
                           const mfem::Array<int>& ess_bdr,
                           const mfem::ParMesh* const pmesh,
                           mfem::ParFiniteElementSpace* const fespace, Result& res);

void spn_slepc_vector(Timer& time, const Input& inp,
                        Material_library& mat_lib,
                        const mfem::Array<int>& ess_bdr,
                        const mfem::ParMesh* const pmesh,
                        mfem::ParFiniteElementSpace* const fespace,
                        Result& res,
                        Result& scalar_flux);

void spn_amgx_vector(Timer& time, const Input& inp,
                     Material_library& mat_lib,
                     const mfem::Array<int>& ess_bdr,
                     const mfem::ParMesh* const pmesh,
                     mfem::ParFiniteElementSpace* const fespace,
                     Result& res);

void spn_slepc_manual(Timer& time, const Input& inp,
                              Material_library& mat_lib,
                              const mfem::Array<int>& ess_bdr,
                              const mfem::ParMesh* const pmesh,
                              mfem::ParFiniteElementSpace* const fespace, Result& res);

void epdo_power_iteration(const Quadrature* const quad, Timer& time, const Input& inp,
                          const Material_library& mat_lib,
                          const mfem::Array<int>& ess_bdr,
                          mfem::ParMesh* const pmesh,
                          mfem::ParFiniteElementSpace* const fespace,
                          Result& angular_flux, Result& scalar_flux);

void sfent_power_iteration(const Quadrature* const quad, Timer& time, const Input& inp,
                           const Material_library& mat_lib,
                           const mfem::Array<int>& ess_bdr,
                           mfem::ParMesh* const pmesh,
                           mfem::ParFiniteElementSpace* const fespace,
                           Result& angular_flux, Result& scalar_flux);

void saaf_power_iteration(const Quadrature* const quad,
                          Timer& time,
                          const Input& inp,
                          const Material_library& mat_lib,
                          mfem::ParMesh* const pmesh,
                          mfem::ParFiniteElementSpace* const fespace,
                          Result& angular_flux,
                          Result& scalar_flux);
void saaf_angle_power_iteration(const Quadrature* const quad,
                                Timer& time,
                                const Input& inp,
                                const Material_library& mat_lib,
                                mfem::ParMesh* const pmesh,
                                Result& angular_flux,
                                Result& scalar_flux);

void saaf_slepc_vector(const Quadrature* const quad,
                       Timer& time,
                       const Input& inp,
                       const Material_library& mat_lib,
                       mfem::ParMesh* const pmesh,
                       mfem::ParFiniteElementSpace* const fespace,
                       Result& angular_flux,
                       Result& scalar_flux);
void saaf_true_slepc_vector(const Quadrature* const quad,
                            Timer& time,
                            const Input& inp,
                            const Material_library& mat_lib,
                            Result& angular_flux);

void saaf_true_amgx_vector(const Quadrature* const quad,
                           Timer& time,
                           const Input& inp,
                           const Material_library& mat_lib,
                           Result& angular_flux);

void saaf_split_slepc_vector(const Quadrature* const quad,
                             Timer& time,
                             const Input& inp,
                             const Material_library& mat_lib,
                             Result& angular_flux,
                             Result& scalar_flux);
void saaf_split_power_iteration(const Quadrature* const quad,
                                Timer& time,
                                const Input& inp,
                                const Material_library& mat_lib,
                                Result& angular_flux,
                                Result& scalar_flux);

/*
void saaf_split_slepc_vector(const Quadrature* const quad,
                             Timer& time,
                             const Input& inp,
                             const Material_library& mat_lib,
                             Result& angular_flux,
                             Result& scalar_flux);
*/
void saaf_self_power_iteration(const Quadrature* const quad,
                               Timer& time,
                               const Input& inp,
                               const Material_library& mat_lib,
                               Result& angular_flux,
                               Result& scalar_flux);

void saaf_true_slepc_diffusion_vector(const Quadrature* const quad,
                                      Timer& time,
                                      const Input& inp,
                                      Material_library& mat_lib,
                                      Result& angular_flux,
                                      Result& scalar_flux);

void saaf_true_slepc_transport_vector(const Quadrature* const quad,
                                      Timer& time,
                                      const Input& inp,
                                      const Material_library& mat_lib,
                                      Result& angular_flux);

void slepc_davidson_hypre_boomeramg_initialize(const Input& inp, int num_equations);

void calc_scalar_flux(int g,
                      const Result& angular_flux, 
                      const Quadrature* const quad,
                      Result& scalar_flux);
void calc_scalar_flux_vector(int g,
                             const Result& angular_flux, 
                             const Quadrature* const quad,
                             Result& scalar_flux);
void calc_scalar_flux_slepc_vector(const Result& angular_flux, 
                                   const Quadrature* const quad,
                                   const Material_library& mat_lib,
                                   Result& scalar_flux);

void calc_power(const Material_library& mat_lib,
                const Result& scalar_flux,
                double rxpwr,
                mfem::ParGridFunction& power);

void calc_fiss_upscatter(const Result& scalar_flux,
                         const Material_library& mat_lib,
                         std::vector<mfem::QuadratureFunction>& fiss_quad_func,
                         std::vector<mfem::QuadratureFunction>& upscatter_quad_func,
                         const bool flux_free_chi=false);
void calc_downscatter(const int g, 
                      const Result& scalar_flux,
                      const Material_library& mat_lib,
                      mfem::QuadratureFunction& dnscatter_quad_func);
void calc_selfscatter(const int g, 
                      const mfem::ParGridFunction &scalar_flux,
                      const Material_library& mat_lib,
                      mfem::QuadratureFunction& selfscatter_quad_func);
void calc_fiss_src(const Result& scalar_flux, 
                   const Material_library& mat_lib,
                   mfem::QuadratureFunction& fiss_src);

void calc_spn_next(const Result& flux,
                   const Material_library& mat_lib,
                   SPNWrapper<mfem::QuadratureFunction>& spn_next_quad_func);
void calc_spn_prev(const int g, const int n,
                   const Result& flux,
                   const Material_library& mat_lib,
                   mfem::QuadratureFunction& spn_prev_quad_func);

void calc_all_scatter(const Result& scalar_flux,
                      const Material_library& mat_lib,
                      const Quadrature* quad,
                      mfem::QuadratureFunction& scat_quad,
                      mfem::QuadratureFunction& scat_saaf_quad,
                      bool include_self_scatter = true);

class Failed_to_converge : std::exception
{
  public:
    Failed_to_converge () {}
    Failed_to_converge (const std::string& message) : 
      msg{"Failed to converge.\n" + message} {}

    const char* what() const noexcept override
    {
      return msg.c_str();
    }

  private:
    const std::string msg;
};

double par_norml2(const mfem::Vector& x);
void flux_sign_fixup(mfem::Vector& pgf);

void power_iteration_amgx(mfem::HypreParMatrix* A, mfem::HypreParMatrix* M, const Input& inp, Timer& time, Result& res, int numfunctions=1);
void power_iteration_hypre(mfem::HypreParMatrix* A, mfem::HypreParMatrix* M, const Input& inp, Timer& time, Result& res, int numfunctions=1);

void slepc_initialize(const Input& inp);
void slepc_solver_wrapper(mfem::HypreParMatrix* A, mfem::HypreParMatrix* M, const Input& inp, Timer& time, Result& res, int numfunctions=1);
void slepc_finalize();

mfem::HypreParMatrix* spn_a_matrix(const Input &inp, 
                                   const Material_library &mat_lib,
                                   mfem::ParFiniteElementSpace* fespace,
                                   const mfem::Array<int>& ess_bdr);
mfem::HypreParMatrix* spn_m_matrix(const Input &inp, 
                                   const Material_library &mat_lib,
                                   mfem::ParFiniteElementSpace* fespace,
                                   const mfem::Array<int>& ess_bdr);

mfem::HypreParMatrix* saaf_a_matrix(const Input &inp, 
                                   const Material_library &mat_lib,
                                   const Quadrature* const quad,
                                   mfem::ParFiniteElementSpace* fespace);
mfem::HypreParMatrix* saaf_m_matrix(const Input &inp, 
                                   const Material_library &mat_lib,
                                   const Quadrature* const quad,
                                   mfem::ParFiniteElementSpace* fespace);

void dg_spn_power_iteration(Input &inp,
                            const Material_library &mat_lib,
                            mfem::ParMesh *pmesh);
void dg_transport_power_iteration(Input &inp,
                                  const Material_library &mat_lib,
                                  const Quadrature *quad,
                                  mfem::ParMesh *pmesh);

void dg_calc_scalar_flux(std::vector<std::unique_ptr<mfem::ParGridFunction>> &angular_flux,
                         const Quadrature *quad,
                         mfem::ParGridFunction &scalar_flux);

void spn_partial_assembly_power_iteration(const Input &inp,
                                          const Material_library &mat_lib,
                                          mfem::Array<int> &ess_bdr,
                                          Result &scalar_flux);

#endif
