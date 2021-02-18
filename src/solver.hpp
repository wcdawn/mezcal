#ifndef SOLVER
#define SOLVER

#include "input.hpp"
#include "material_library.hpp"
#include "mfem.hpp"
#include "result.hpp"
#include "timer.hpp"
#include "xslibrary.hpp"
#include "quadrature.hpp"

#include <exception>
#include <string>

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

void diffusion_fixed_src(Timer& time, const Input& inp,
                           const Material_library& mat_lib,
                           const mfem::Array<int>& ess_bdr,
                           const mfem::ParMesh* const pmesh,
                           mfem::ParFiniteElementSpace* const fespace, Result& res);

void spn_slepc_vector(Timer& time, const Input& inp,
                        Material_library& mat_lib,
                        const mfem::Array<int>& ess_bdr,
                        const mfem::ParMesh* const pmesh,
                        mfem::ParFiniteElementSpace* const fespace, Result& res);

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
                                mfem::ParFiniteElementSpace* const fespace,
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
                            mfem::ParMesh* const pmesh,
                            mfem::ParFiniteElementSpace* const fespace,
                            Result& angular_flux,
                            Result& scalar_flux);

void slepc_davidson_hypre_boomeramg_initialize(const Input& inp, int num_equations);

void calc_scalar_flux(int g, const Result& angular_flux, const Quadrature* const quad, Result& scalar_flux);

void calc_fiss_upscatter(const Result& scalar_flux,
                         const Material_library& mat_lib,
                         std::vector<mfem::QuadratureFunction>& fiss_quad_func,
                         std::vector<mfem::QuadratureFunction>& upscatter_quad_func,
                         const bool flux_free_chi=false);
void calc_downscatter(const int g, 
                      const Result& scalar_flux,
                      const Material_library& mat_lib,
                      mfem::QuadratureFunction& dnscatter_quad_func);

void calc_spn_next(const Result& flux,
                   const Material_library& mat_lib,
                   SPNWrapper<mfem::QuadratureFunction>& spn_next_quad_func);
void calc_spn_prev(const int g, const int n,
                   const Result& flux,
                   const Material_library& mat_lib,
                   mfem::QuadratureFunction& spn_prev_quad_func);

void calc_fiss_src(const Result& scalar_flux, 
                      const Material_library& mat_lib,
                      mfem::QuadratureFunction& fiss_src);

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

#endif
