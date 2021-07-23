#ifndef OUPTUT_HPP
#define OUTPUT_HPP

#include "mfem.hpp"

#include "input.hpp"
#include "material_library.hpp"
#include "result.hpp"

#include <string>
#include <iostream>

void total_xs_plot(mfem::ParMesh *pmesh,
                   const Material_library &mat_lib,
                   const std::string out_name);

void coarse_energy_plot(mfem::ParMesh *pmesh,
                        const Result &scalar_flux,
                        const std::vector<int> collapse_vector,
                        const std::string out_name);

void mesh_stats(mfem::ParMesh *pmesh, std::ostream &os = std::cout);

#endif
