#ifndef SOLVER_SPN_POWER_ITERATION_HPP
#define SOLVER_SPN_POWER_ITERATION_HPP

#include "input.hpp"
#include "material_library.hpp"
#include "result.hpp"
#include "timer.hpp"

#include "mfem.hpp"

#include <vector>

void spn_power_iteration(Timer &time,
                         Input &inp,
                         Material_library &mat_lib,
                         std::vector<std::unique_ptr<mfem::ParGridFunction>> &phi,
                         double &keff);

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

#endif
