#ifndef SOLVER_SAAF_ANGLE_POWER_ITERATION_HPP
#define SOLVER_SAAF_ANGLE_POWER_ITERATION_HPP

#include "input.hpp"
#include "material_library.hpp"
#include "quadrature.hpp"

#include "result.hpp"
#include "timer.hpp"

#include "mfem.hpp"

#include <vector>

void saaf_angle_power_iteration(const Quadrature * const quad,
                                Timer &time,
                                Input &inp,
                                const Material_library &mat_lib,
                                std::vector<std::unique_ptr<mfem::ParGridFunction>> &angular_flux,
                                std::vector<std::unique_ptr<mfem::ParGridFunction>> &scalar_flux,
                                double &keff,
                                mfem::ParGridFunction &power,
                                mfem::ParGridFunction &temperature);

void calc_fiss_upscatter_vector(const std::vector<std::unique_ptr<mfem::ParGridFunction>> &scalar_flux,
                                const Material_library &mat_lib,
                                const Quadrature * const quad,
                                std::vector<mfem::QuadratureFunction> &fiss_upscatter_quad,
                                std::vector<mfem::QuadratureFunction> &saaf_fiss_upscatter_quad,
                                const double keff,
                                const bool flux_free_chi=false);

void calc_downscatter_vector(const int g, 
                             const std::vector<std::unique_ptr<mfem::ParGridFunction>> &scalar_flux,
                             const Material_library &mat_lib,
                             const Quadrature * const quad,
                             mfem::QuadratureFunction &dnscatter_quad,
                             mfem::QuadratureFunction &saaf_dnscatter_quad);

void calc_selfscatter_vector(const int g, 
                             const mfem::ParGridFunction &scalar_flux,
                             const Material_library &mat_lib,
                             const Quadrature * const quad,
                             mfem::QuadratureFunction &selfscatter_quad,
                             mfem::QuadratureFunction &saaf_selfscatter_quad);

void calc_scalar_flux_vector(const mfem::ParGridFunction &angular_flux, 
                             const Quadrature* const quad,
                             mfem::ParGridFunction &scalar_flux);

/* 
 * WARNING
 * This is *NO LONGER A DUPLICATE* of the function of the same name from in solver.[ch]pp.
 * WARNING
 */
static void calc_fiss_src(const std::vector<std::unique_ptr<mfem::ParGridFunction>> &scalar_flux, 
                   const Material_library &mat_lib,
                   mfem::QuadratureFunction &fiss_src);

void calc_power_vector(const Material_library &mat_lib,
                       const std::vector<std::unique_ptr<mfem::ParGridFunction>> &scalar_flux,
                       const double rxpwr,
                       mfem::QuadratureFunction &power_quad);

#endif
