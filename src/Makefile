# Makefile

ifeq ($(MACH),SUMMIT)
STD = c++11
MPICXX = mpixlC
H5_DIR = $(OLCF_HDF5_ROOT)/lib

PROJ = /ccs/proj/csc404
MFEM_DIR = $(PROJ)/mfem
HYPRE_DIR = $(PROJ)/hypre-2.16.0/src/hypre
METIS_DIR = $(OLCF_METIS_ROOT)
METIS_LIB = -L$(METIS_DIR)/lib -l:libmetis.so

# CUDA_DIR is set automatically by the module

PETSC_DIR = $(PROJ)/petsc/arch-olcf-summit-opt
SLEPC_DIR = $(PROJ)/slepc/arch-olcf-summit-opt

else ifeq ($(MACH),ARCHBUILD)
STD = c++17
MPICXX = /opt/mpich/bin/mpicxx
H5_DIR = /usr/lib

MFEM_DIR = ../build/mfem
HYPRE_DIR = ../hypre/src/hypre
METIS_DIR = ../metis

METIS_LIB = $(METIS_DIR)/build/Linux-x86_64/libmetis/libmetis.a

CUDA_DIR = /opt/cuda
PETSC_DIR = /home/wcdawn/projects/petsc/arch-linux-c-debug
SLEPC_DIR = /home/wcdawn/projects/slepc/arch-linux-c-debug

else

$(error Invalid MACH)

endif

CXXFLAGS = -Wall -pedantic-errors -std=$(STD)

MPICXXFLAGS = $(CXXFLAGS)

H5CXXFLAGS = $(CXXFLAGS)
# generated using "h5c++ -show"
H5_LIB = -L$(OLCF_HDF5_ROOT) -lhdf5_hl_cpp -lhdf5_cpp -lhdf5_hl -lhdf5 -L$(H5_DIR) -lz

LD = $(MPICXX) 
LDFLAGS = -std=$(STD)

PROJ = /ccs/proj/csc404

MFEM_INCLUDE = -I$(MFEM_DIR)
MFEM_HEADER = $(MFEM_DIR)/mfem.hpp
MFEM_LIB = $(MFEM_DIR)/libmfem.a

HYPRE_INCLUDE = -I$(HYPRE_DIR)/include
HYPRE_LIB = $(HYPRE_DIR)/lib/libHYPRE.a

METIS_INCLUDE = -I$(METIS_DIR)/include

CUDA_INCLUDE = -I$(CUDA_DIR)/include
CUDA_LIB = -L$(CUDA_DIR)/lib64 -lcudart_static -ldl -lrt -pthread -lcusparse_static

PETSC_INCLUDE = -I$(PETSC_DIR)/include -I$(PETSC_DIR)/../include
PETSC_LIB = -L$(PETSC_DIR)/lib -l:libpetsc.so

SLEPC_INCLUDE = -I$(SLEPC_DIR)/include -I$(SLEPC_DIR)/../include
SLEPC_LIB = -L$(SLEPC_DIR)/lib -l:libslepc.so

INCLUDE_DIR = ../include

INCLUDE = $(MFEM_INCLUDE) $(HYPRE_INCLUDE) $(METIS_INCLUDE) $(CUDA_INCLUDE) $(PETSC_INCLUDE) $(SLEPC_INCLUDE) -I$(INCLUDE_DIR)
LIBS = $(MFEM_LIB) $(HYPRE_LIB) $(METIS_LIB) $(CUDA_LIB) $(H5_LIB) $(PETSC_LIB) $(SLEPC_LIB)

ifeq ($(METHOD), dbg)
CXXFLAGS += -O0 -g
else
CXXFLAGS += -O3
endif

CPP_FILES = main.cpp \
			analytic.cpp \
			boundary_abs_normal_bf_integrator.cpp \
			fission_coefficient.cpp \
			hdf5_access_wrapper.cpp \
			input.cpp \
			legendre_chebyschev_quadrature.cpp \
			level_symmetric_quadrature.cpp \
			material_library.cpp \
			matrix_pwconst_coefficient.cpp \
			quadrature.cpp \
			result.cpp \
			scattering_coefficient.cpp \
			solver.cpp \
			solver_choice.cpp \
			spn_source_coefficient.cpp \
			timer.cpp \
			vector_pwconst_coefficient.cpp \
			version.cpp \
			xslibrary.cpp

HPP_FILES = $(INCLUDE_DIR)/analytic.hpp \
			$(INCLUDE_DIR)/boundary_abs_normal_bf_integrator.hpp \
			$(INCLUDE_DIR)/fission_coefficient.hpp \
			$(INCLUDE_DIR)/hdf5_access_wrapper.hpp \
			$(INCLUDE_DIR)/input.hpp \
			$(INCLUDE_DIR)/legendre_chebyschev_quadrature.hpp \
			$(INCLUDE_DIR)/level_symmetric_quadrature.hpp \
			$(INCLUDE_DIR)/make_unique.hpp \
			$(INCLUDE_DIR)/material_library.hpp \
			$(INCLUDE_DIR)/matrix_pwconst_coefficient.hpp \
			$(INCLUDE_DIR)/quadrature.hpp \
			$(INCLUDE_DIR)/scattering_coefficient.hpp \
			$(INCLUDE_DIR)/solver.hpp \
			$(INCLUDE_DIR)/solver_choice.hpp \
			$(INCLUDE_DIR)/spn_source_coefficient.hpp \
			$(INCLUDE_DIR)/timer.hpp \
			$(INCLUDE_DIR)/vector_pwconst_coefficient.hpp \
			$(INCLUDE_DIR)/xslibrary.hpp

OBJ = $(CPP_FILES:.cpp=.o)

EXEC = micro.x

GIT_VERSION = $(shell git describe --dirty --always --tags)
GIT_SHA     = $(shell git log -n1 --format=format:"%H")
DATE        = $(shell date +"%a %d %b %Y %T %Z") # TODO replace T with R
HOST        = $(shell hostname)

.SUFFIXES :
.SUFFIXES : .cpp .o .hpp

.PHONY : default all clean version

default : all

all : $(EXEC)

$(EXEC) : $(OBJ)
	@# delete any existing executable if linking fails
	@$(MAKE) version -s
	$(LD) $(LDFLAGS) $^ -o $@ $(LIBS) || $(RM) $(EXEC)

%.o : %.cpp
	$(MPICXX) $(MPICXXFLAGS) $(INCLUDE) -c $<

version:
	$(RM) $(INCLUDE_DIR)/version.hpp
	$(MAKE) $(INCLUDE_DIR)/version.hpp
	$(MAKE) version.o

version.o : version.cpp $(INCLUDE_DIR)/version.hpp

$(INCLUDE_DIR)/version.hpp :
	@echo "#ifndef VERSION" > ../include/version.hpp
	@echo "#define VERSION" >> ../include/version.hpp
	@echo "" >> ../include/version.hpp
	@echo "#include <string>" >> ../include/version.hpp
	@echo "" >> ../include/version.hpp
	@echo "namespace Version" >> ../include/version.hpp
	@echo "{" >> ../include/version.hpp
	@echo "  const std::string git_version   = \"$(GIT_VERSION)\";" >> ../include/version.hpp
	@echo "  const std::string git_sha       = \"$(GIT_SHA)\";" >> ../include/version.hpp
	@echo "  const std::string compiled_date = \"$(DATE)\";" >> ../include/version.hpp
	@echo "  const std::string compiled_host = \"$(HOST)\";" >> ../include/version.hpp
	@echo "} // namespace Version" >> ../include/version.hpp
	@echo "" >> ../include/version.hpp
	@echo "#endif" >> ../include/version.hpp

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
	@$(RM) $(OBJ) $(EXEC) version.cpp $(INCLUDE_DIR)/version.hpp

format : $(CPP_FILES) $(HPP_FILES)
	@clang-format -i --verbose $^

# Listing Order:
# <object file> : $(MFEM_HEADER) <my header> <other headers>
main.o : $(MFEM_HEADER) $(INCLUDE_DIR)/analytic.hpp $(INCLUDE_DIR)/input.hpp $(INCLUDE_DIR)/material_library.hpp $(INCLUDE_DIR)/result.hpp $(INCLUDE_DIR)/solver.hpp $(INCLUDE_DIR)/timer.hpp $(INCLUDE_DIR)/xslibrary.hpp
analytic.o : $(MFEM_HEADER) $(INCLUDE_DIR)/analytic.hpp $(INCLUDE_DIR)/result.hpp $(INCLUDE_DIR)/xslibrary.hpp
xslibrary.o : $(MFEM_HEADER) $(INCLUDE_DIR)/xslibrary.hpp
input.o : $(MFEM_HEADER) $(INCLUDE_DIR)/analytic.hpp $(INCLUDE_DIR)/input.hpp $(INCLUDE_DIR)/xslibrary.hpp
material_library.o : $(MFEM_HEADER) $(INCLUDE_DIR)/material_library.hpp $(INCLUDE_DIR)/xslibrary.hpp
matrix_pwconst_coefficient.o : $(MFEM_HEADER) $(INCLUDE_DIR)/matrix_pwconst_coefficient.hpp
solver.o : $(MFEM_HEADER) $(INCLUDE_DIR)/solver.hpp $(INCLUDE_DIR)/input.hpp $(INCLUDE_DIR)/spn_source_coefficient.hpp $(INCLUDE_DIR)/fission_coefficient.hpp $(INCLUDE_DIR)/material_library.hpp $(INCLUDE_DIR)/scattering_coefficient.hpp $(INCLUDE_DIR)/timer.hpp $(INCLUDE_DIR)/xslibrary.hpp $(INCLUDE_DIR)/result.hpp $(INCLUDE_DIR)/boundary_abs_normal_bf_integrator.hpp
scattering_coefficient.o : $(MFEM_HEADER) $(INCLUDE_DIR)/scattering_coefficient.hpp $(INCLUDE_DIR)/material_library.hpp $(INCLUDE_DIR)/xslibrary.hpp $(INCLUDE_DIR)/result.hpp
fission_coefficient.o : $(MFEM_HEADER) $(INCLUDE_DIR)/fission_coefficient.hpp $(INCLUDE_DIR)/material_library.hpp $(INCLUDE_DIR)/xslibrary.hpp $(INCLUDE_DIR)/result.hpp
hdf5_access_wrapper.o : $(INCLUDE_DIR)/hdf5_access_wrapper.hpp
timer.o : $(INCLUDE_DIR)/timer.hpp
version.o : $(INCLUDE_DIR)/version.hpp
vector_pwconst_coefficient.o : $(MFEM_HEADER) $(INCLUDE_DIR)/vector_pwconst_coefficient.hpp
spn_source_coefficient.o : $(MFEM_HEADER) $(INCLUDE_DIR)/spn_source_coefficient.hpp $(INCLUDE_DIR)/result.hpp $(INCLUDE_DIR)/material_library.hpp $(INCLUDE_DIR)/xslibrary.hpp
solver_choice.o : $(INCLUDE_DIR)/solver_choice.hpp
result.o : $(MFEM_HEADER) $(INCLUDE_DIR)/result.hpp
legendre_chebyschev_quadrature.o : $(MFEM_HEADER) $(INCLUDE_DIR)/legendre_chebyschev_quadrature.hpp $(INCLUDE_DIR)/quadrature.hpp
level_symmetric_quadrature.o : $(MFEM_HEADER) $(INCLUDE_DIR)/level_symmetric_quadrature.hpp $(INCLUDE_DIR)/quadrature.hpp
quadrature.o : $(MFEM_HEADER) $(INCLUDE_DIR)/quadrature.hpp
boundary_abs_normal_bf_integrator.o : $(MFEM_HEADER) $(INCLUDE_DIR)/boundary_abs_normal_bf_integrator.hpp
