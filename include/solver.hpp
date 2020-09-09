#ifndef SOLVER
#define SOLVER

#include "input.hpp"
#include "material_library.hpp"
#include "mfem.hpp"
#include "result.hpp"
#include "timer.hpp"
#include "xslibrary.hpp"

#include <exception>
#include <string>

void diffusion_hypre_lobpcg(Timer& time, const Input& inp,
                              const Material_library& mat_lib,
                              const mfem::Array<int>& ess_bdr,
                              const mfem::ParMesh* const pmesh,
                              mfem::ParFiniteElementSpace* const fespace, Result& res);

void diffusion_power_iteration(Timer& time, const Input& inp,
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

void diffusion_slepc_manual(Timer& time, const Input& inp,
                              const Material_library& mat_lib,
                              const mfem::Array<int>& ess_bdr,
                              const mfem::ParMesh* const pmesh,
                              mfem::ParFiniteElementSpace* const fespace, Result& res);

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

private:

  const int G;
  const int N;
  std::vector<T> arr;

};

#endif
