#ifndef CONDUCTION_HPP
#define CONDUCTION_HPP

#include "material_library.hpp"
#include "input.hpp"

#include "mfem.hpp"

void temperature_heat_conduction(const Input& inp,
                                 const Material_library& mat_lib, 
                                 const mfem::Array<int>& ess_bdr,
                                 const mfem::ParGridFunction& power,
                                 mfem::ParGridFunction& temperature);

#endif
