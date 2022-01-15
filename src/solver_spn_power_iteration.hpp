#ifndef SOLVER_SPN_POWER_ITERATION_HPP
#define SOLVER_SPN_POWER_ITERATION_HPP

#include "input.hpp"
#include "material_library.hpp"
#include "result.hpp"
#include "timer.hpp"

#include "mfem.hpp"

#include <vector>

constexpr double spn_bc_dat_0[4][4] =
{
   -3.0/2.0   ,  -7.0/12.0 ,  -11.0/30.0  , -15.0/56.0 ,
  -15.0/8.0   ,  35.0/12.0 ,   55.0/48.0  ,   3.0/4.0  ,
    9.0/16.0  , 189.0/64.0 , -297.0/80.0  , -45.0/32.0 ,
  -39.0/128.0 , -91.0/96.0 , -715.0/192.0 , 975.0/224.0
};

constexpr double spn_bc_dat_1[4][3] =
{
     7.0/24.0  ,  11.0/60.0  ,   15.0/112.0 ,
   -35.0/24.0  , -55.0/96.0  ,   -3.0/8.0   ,
  -189.0/128.0 , 297.0/160.0 ,   45.0/64.0  ,
    91.0/192.0 , 715.0/384.0 , -975.0/448.0
};

constexpr double spn_bc_dat_2[4][2] =
{
   -11.0/80.0  ,   -45.0/448.0 ,
    55.0/128.0 ,     9.0/32.0  ,
  -891.0/640.0 ,  -135.0/256.0 ,
  -715.0/512.0 ,  2925.0/1792.0
};

constexpr double spn_bc_dat_3[4][1] =
{
    75.0/896.0  ,
   -15.0/64.0   ,
   225.0/512.0  ,
  4875.0/3584.0
};

const mfem::DenseMatrix spn_bc_mat_0(spn_bc_dat_0);
const mfem::DenseMatrix spn_bc_mat_1(spn_bc_dat_1);
const mfem::DenseMatrix spn_bc_mat_2(spn_bc_dat_2);
const mfem::DenseMatrix spn_bc_mat_3(spn_bc_dat_3);

const std::array<mfem::DenseMatrix,4> spn_bc_mat =
{
  spn_bc_mat_0,
  spn_bc_mat_1,
  spn_bc_mat_2,
  spn_bc_mat_3
};

void spn_power_iteration(Timer &time,
                         Input &inp,
                         Material_library &mat_lib,
                         std::vector<std::unique_ptr<mfem::ParGridFunction>> &phi,
                         double &keff,
                         mfem::ParGridFunction &power,
                         mfem::ParGridFunction &temperature);

void spn_calc_fiss_upscatter(const std::vector<std::unique_ptr<mfem::ParGridFunction>> &phi,
                             double keff,
                             const Material_library &mat_lib,
                             std::vector<mfem::QuadratureFunction> &fiss_upscatter_quad,
                             bool flux_free_chi = false);


void spn_calc_spn_next(const std::vector<std::unique_ptr<mfem::ParGridFunction>> &phi,
                       int spn_eq,
                       const Material_library &mat_lib,
                       std::vector<mfem::QuadratureFunction> &spn_next_quad);

void spn_calc_downscatter(int g,
                          const std::vector<std::unique_ptr<mfem::ParGridFunction>> &phi,
                          const Material_library &mat_lib,
                          mfem::QuadratureFunction &dnscatter_quad);

void spn_calc_spn_prev(int g, int n,
                       const std::vector<std::unique_ptr<mfem::ParGridFunction>> &phi,
                       const Material_library &mat_lib,
                       mfem::QuadratureFunction &dnscatter_quad);

void spn_calc_fiss_src(const std::vector<std::unique_ptr<mfem::ParGridFunction>> &phi,
                       const Material_library &mat_lib,
                       mfem::QuadratureFunction &fiss_src_quad);

void spn_calc_power(const Material_library &mat_lib,
                    const std::vector<std::unique_ptr<mfem::ParGridFunction>> &phi,
                    const double rxpwr,
                    mfem::QuadratureFunction &power_quad);

#endif
