# Makefile

include ../Makefile.include

ADDON_DIR = ../addon

CPP_FILES = main.cpp \
			analytic.cpp \
			conduction.cpp \
			delaunay.cpp \
			fission_coefficient.cpp \
			hdf5_access_wrapper.cpp \
			incoming_flux_coefficient.cpp \
			input.cpp \
			legendre_chebyschev_quadrature.cpp \
			level_symmetric_quadrature.cpp \
			material_library.cpp \
			matrix_pwconst_coefficient.cpp \
			nda_boundary_factor_coefficient.cpp \
			nda_diffusion_coefficient.cpp \
			output.cpp \
			quadrature.cpp \
			result.cpp \
			saaf_void_tau_coefficient.cpp \
			scattering_coefficient.cpp \
			solver.cpp \
			solver_choice.cpp \
			spn_source_coefficient.cpp \
			stability_tau_coefficient.cpp \
			thermal_conductivity_coefficient.cpp \
			timer.cpp \
			vector_integrator_wrapper.cpp \
			vector_product_coefficient.cpp \
			vector_pwconst_coefficient.cpp \
			version.cpp \
			xslibrary.cpp \
			upscattering_coefficient.cpp \
			downscattering_coefficient.cpp \
			$(ADDON_DIR)/bilininteg.cpp \
			$(ADDON_DIR)/lininteg.cpp

HPP_FILES = analytic.hpp \
			conduction.hpp \
			delaunay.hpp \
			fission_coefficient.hpp \
			hdf5_access_wrapper.hpp \
			incoming_flux_coefficient.hpp \
			input.hpp \
			legendre_chebyschev_quadrature.hpp \
			level_symmetric_quadrature.hpp \
			make_unique.hpp \
			material_library.hpp \
			matrix_pwconst_coefficient.hpp \
			memory.hpp \
			nda_boundary_factor_coefficient.hpp \
			nda_diffusion_coefficient.hpp \
			output.hpp \
			quadrature.hpp \
			result.hpp \
			saaf_void_tau_coefficient.hpp \
			scattering_coefficient.hpp \
			solver.hpp \
			solver_choice.hpp \
			spn_source_coefficient.hpp \
			stability_tau_coefficient.hpp \
			thermal_conductivity_coefficient.hpp \
			timer.hpp \
			vector_integrator_wrapper.hpp \
			vector_product_coefficient.hpp \
			vector_pwconst_coefficient.hpp \
			xslibrary.hpp \
			upscattering_coefficient.hpp \
			downscattering_coefficient.hpp \
			$(ADDON_DIR)/bilininteg.cpp \
			$(ADDON_DIR)/lininteg.cpp \
			$(ADDON_DIR)/tensor.hpp \
			$(ADDON_DIR)/mfem_addon.hpp

OBJ = $(CPP_FILES:.cpp=.o)

EXEC = micro.x

GIT_VERSION = $(shell git describe --dirty --always --tags)
GIT_SHA     = $(shell git log -n1 --format=format:"%H")
DATE        = $(shell date +"%a %d %b %Y %T %Z")
HOST        = $(shell hostname)

.SUFFIXES :
.SUFFIXES : .cpp .o .hpp

.PHONY : default all clean version

default : all

all : $(EXEC)

link :
	$(RM) $(EXEC)
	$(MAKE) $(EXEC)

$(EXEC) : $(OBJ)
	@$(MAKE) version -s
	@$(MAKE) -C $(ADDON_DIR)
	$(NVCC) $(CUDA_ARCH) $(LDFLAGS) -Xcompiler -fopenmp -Xcompiler -pthread $^ -o $@ $(LIBS) $(MPI_LIB)

%.o : %.cpp
	$(MPICXX) $(MPICXXFLAGS) $(INCLUDE) -c $<

version:
	$(RM) version.hpp
	$(MAKE) version.hpp
	$(MAKE) version.o

version.o : version.cpp version.hpp

version.hpp :
	@echo "#ifndef VERSION" > version.hpp
	@echo "#define VERSION" >> version.hpp
	@echo "" >> ../include/version.hpp
	@echo "#include <string>" >> version.hpp
	@echo "" >> version.hpp
	@echo "namespace Version" >> version.hpp
	@echo "{" >> version.hpp
	@echo "  const std::string git_version   = \"$(GIT_VERSION)\";" >> version.hpp
	@echo "  const std::string git_sha       = \"$(GIT_SHA)\";" >> version.hpp
	@echo "  const std::string compiled_date = \"$(DATE)\";" >> version.hpp
	@echo "  const std::string compiled_host = \"$(HOST)\";" >> version.hpp
	@echo "} // namespace Version" >> version.hpp
	@echo "" >> version.hpp
	@echo "#endif" >> version.hpp

version.cpp :
	@echo "#include \"version.hpp\"" > version.cpp
	@echo "#include <iostream>" >> version.cpp
	@echo "" >> version.cpp
	@echo "void version(std::ostream& os)" >> version.cpp
	@echo "{" >> version.cpp
	@echo "  os << \"git version   = \" << Version::git_version   << \"\\n\";" >> version.cpp
	@echo "  os << \"git sha       = \" << Version::git_sha       << \"\\n\";" >> version.cpp
	@echo "  os << \"compiled date = \" << Version::compiled_date << \"\\n\";" >> version.cpp
	@echo "  os << \"compiled on   = \" << Version::compiled_host << \"\\n\";" >> version.cpp
	@echo "}" >> version.cpp

clean :
	$(RM) $(OBJ) $(EXEC) version.cpp version.hpp
	$(MAKE) -C $(ADDON_DIR) clean

format : $(CPP_FILES) $(HPP_FILES)
	@clang-format -i --verbose $^

addon :
	$(MAKE) -C $(ADDON_DIR)

# Listing Order:
# <object file> : $(MFEM_HEADER) <my header> <other headers>
main.o : $(MFEM_HEADER) analytic.hpp input.hpp material_library.hpp result.hpp solver.hpp timer.hpp xslibrary.hpp quadrature.hpp legendre_chebyschev_quadrature.hpp level_symmetric_quadrature.hpp output.hpp
analytic.o : $(MFEM_HEADER) analytic.hpp result.hpp xslibrary.hpp
xslibrary.o : $(MFEM_HEADER) xslibrary.hpp
input.o : $(MFEM_HEADER) analytic.hpp input.hpp xslibrary.hpp quadrature.hpp
material_library.o : $(MFEM_HEADER) material_library.hpp xslibrary.hpp
matrix_pwconst_coefficient.o : $(MFEM_HEADER) matrix_pwconst_coefficient.hpp
solver.o : $(MFEM_HEADER) solver.hpp input.hpp spn_source_coefficient.hpp fission_coefficient.hpp material_library.hpp scattering_coefficient.hpp timer.hpp xslibrary.hpp result.hpp quadrature.hpp vector_integrator_wrapper.hpp tensor.hpp memory.hpp upscattering_coefficient.hpp downscattering_coefficient.hpp vector_product_coefficient.hpp nda_boundary_factor_coefficient.hpp nda_diffusion_coefficient.hpp conduction.hpp
scattering_coefficient.o : $(MFEM_HEADER) scattering_coefficient.hpp material_library.hpp xslibrary.hpp result.hpp
fission_coefficient.o : $(MFEM_HEADER) fission_coefficient.hpp material_library.hpp xslibrary.hpp result.hpp
hdf5_access_wrapper.o : hdf5_access_wrapper.hpp
timer.o : timer.hpp
version.o : version.hpp
vector_pwconst_coefficient.o : $(MFEM_HEADER) vector_pwconst_coefficient.hpp
spn_source_coefficient.o : $(MFEM_HEADER) spn_source_coefficient.hpp result.hpp material_library.hpp xslibrary.hpp
solver_choice.o : solver_choice.hpp
result.o : $(MFEM_HEADER) result.hpp
incoming_flux_coefficient.o : $(MFEM_HEADER) incoming_flux_coefficient.hpp
level_symmetric_quadrature.o : $(MFEM_HEADER) level_symmetric_quadrature.hpp quadrature.hpp
legendre_chebyschev_quadrature.o : $(MFEM_HEADER) legendre_chebyschev_quadrature.hpp quadrature.hpp
quadrature.o : $(MFEM_HEADER) quadrature.hpp delaunay.hpp
delaunay.o : $(MFEM_HEADER) delaunay.hpp
stability_tau_coefficient.o : $(MFEM_HEADER) stability_tau_coefficient.hpp
vector_integrator_wrapper.o : $(MFEM_HEADER) vector_integrator_wrapper.hpp
conduction.o : $(MFEM_HEADER) conduction.hpp material_library.hpp thermal_conductivity_coefficient.hpp
thermal_conductivity_coefficient.o : $(MFEM_HEADER) thermal_conductivity_coefficient.hpp material_library.hpp
upscattering_coefficient.o : $(MFEM_HEADER) upscattering_coefficient.hpp material_library.hpp result.hpp
downscattering_coefficient.o : $(MFEM_HEADER) downscattering_coefficient.hpp material_library.hpp result.hpp
output.o : $(MFEM_HEADER) output.hpp input.hpp material_library.hpp saaf_void_tau_coefficient.hpp result.hpp
saaf_void_tau_coefficient.o : $(MFEM_HEADER) saaf_void_tau_coefficient.hpp material_library.hpp
vector_product_coefficient.o : $(MFEM_HEADER) vector_product_coefficient.hpp
nda_boundary_factor_coefficient.o : $(MFEM_HEADER) nda_boundary_factor_coefficient.hpp quadrature.hpp material_library.hpp
nda_diffusion_coefficient.o : $(MFEM_HEADER) nda_diffusion_coefficient.hpp quadrature.hpp material_library.hpp
