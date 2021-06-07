#ifndef OUPTUT_HPP
#define OUTPUT_HPP

#include "mfem.hpp"

#include "input.hpp"
#include "material_library.hpp"

#include <string>

void total_xs_plot(mfem::ParMesh *pmesh,
                   const Material_library &mat_lib,
                   const std::string out_name);

#endif
