# Makefile

CXX = g++
CXXFLAGS = -Wall -pedantic-errors -std=c++17

MPICXX = /opt/mpich/bin/mpicxx -cxx=$(CXX)
MPICXXFLAGS = $(CXXFLAGS)

LD = $(MPICXX) 
LDFLAGS = -std=c++17

MFEM_DIR = ../build/mfem
MFEM_INCLUDE = $(MFEM_DIR)
MFEM_HEADER = $(MFEM_INCLUDE)/mfem.hpp
MFEM_LIB = $(MFEM_DIR)/libmfem.a

HYPRE_DIR = ../hypre/src/hypre
HYPRE_INCLUDE = $(HYPRE_DIR)/include
HYPRE_LIB = $(HYPRE_DIR)/lib/libHYPRE.a

METIS_DIR = ../metis
METIS_INCLUDE = $(METIS_DIR)/include
METIS_LIB = $(METIS_DIR)/build/Linux-x86_64/libmetis/libmetis.a

CUDA_DIR = /opt/cuda
CUDA_LIB = -L$(CUDA_DIR)/lib64 -lcudart_static -ldl -lrt -pthread

ifeq ($(METHOD), dbg)
CXXFLAGS += -O0 -g
else
CXXFLAGS += -O3
endif

CPP_FILES = main.cpp \
			analytic.cpp \
			fission_coefficient.cpp \
			input.cpp \
			material_library.cpp \
			scattering_coefficient.cpp \
			solver.cpp \
			xslibrary.cpp

INCLUDE_DIR = ../include

HPP_FILES = $(INCLUDE_DIR)/analytic.hpp \
			$(INCLUDE_DIR)/fission_coefficient.hpp \
			$(INCLUDE_DIR)/input.hpp \
			$(INCLUDE_DIR)/material_library.hpp \
			$(INCLUDE_DIR)/scattering_coefficient.hpp \
			$(INCLUDE_DIR)/solver.hpp \
			$(INCLUDE_DIR)/xslibrary.hpp

OBJ = $(CPP_FILES:.cpp=.o)

EXEC = micro.x

GIT_VERSION = $(shell git describe --dirty --always --tags)
GIT_SHA     = $(shell git log -n1 --format=format:"%H")
DATE        = $(shell date +"%a %d %b %Y %R %Z")
HOST        = $(shell hostname)

.SUFFIXES :
.SUFFIXES : .cpp .o .hpp

.PHONY : default all clean

default : all

all : $(EXEC)

$(EXEC) : $(OBJ) $(MFEM_LIB) $(HYPRE_LIB) $(METIS_LIB)
	@# delete any existing executable if linking fails
	$(LD) $(LDFLAGS) $^ -o $@ $(MFEM_LIB) $(HYPRE_LIB) $(METIS_LIB) $(CUDA_LIB) || $(RM) $(EXEC)

%.o : %.cpp
	$(MPICXX) $(MPICXXFLAGS) -I$(INCLUDE_DIR) -I$(MFEM_INCLUDE) -I$(HYPRE_INCLUDE) -I$(METIS_INCLUDE) -c $<

version.hpp : 
	@echo "#ifndef VERSION" > version.hpp
	@echo "#define VERSION" >> version.hpp
	@echo "" >> version.hpp
	@echo "#include <string>" >> version.hpp
	@echo "" >> version.hpp
	@echo "namespace Version" >> version.hpp
	@echo "{" >> version.hpp
	@echo "  const std::string git_version   = \"$(GIT_VERSION)\";" >> version.hpp
	@echo "  const std::string git_sha       = \"$(GIT_SHA)\";" >> version.hpp
	@echo "  const std::string compiled_date = \"$(DATE)\";" >> version.hpp
	@echo "  const std::string compiled_host = \"$(HOST)\";" >> version.hpp
	@echo "}; // namespace Version" >> version.hpp
	@echo "" >> version.hpp
	@echo "#endif" >> version.hpp

clean :
	@$(RM) $(OBJ) $(EXEC) version.hpp

format : $(CPP_FILES) $(HPP_FILES)
	@clang-format -i --verbose $^

# Listing Order:
# <object file> : $(MFEM_HEADER) <my header> <other headers>
main.o : $(MFEM_HEADER) $(INCLUDE_DIR)/analytic.hpp $(INCLUDE_DIR)/input.hpp $(INCLUDE_DIR)/material_library.hpp $(INCLUDE_DIR)/result.hpp $(INCLUDE_DIR)/solver.hpp $(INCLUDE_DIR)/xslibrary.hpp
analytic.o : $(MFEM_HEADER) $(INCLUDE_DIR)/analytic.hpp $(INCLUDE_DIR)/result.hpp $(INCLUDE_DIR)/xslibrary.hpp
xslibrary.o : $(MFEM_HEADER) $(INCLUDE_DIR)/xslibrary.hpp
input.o : $(MFEM_HEADER) $(INCLUDE_DIR)/analytic.hpp $(INCLUDE_DIR)/input.hpp 
material_library.o : $(MFEM_HEADER) $(INCLUDE_DIR)/material_library.hpp $(INCLUDE_DIR)/xslibrary.hpp
solver.o : $(MFEM_HEADER) $(INCLUDE_DIR)/solver.hpp $(INCLUDE_DIR)/input.hpp $(INCLUDE_DIR)/fission_coefficient.hpp $(INCLUDE_DIR)/material_library.hpp $(INCLUDE_DIR)/scattering_coefficient.hpp $(INCLUDE_DIR)/xslibrary.hpp $(INCLUDE_DIR)/result.hpp
scattering_coefficient.o : $(MFEM_HEADER) $(INCLUDE_DIR)/scattering_coefficient.hpp $(INCLUDE_DIR)/material_library.hpp $(INCLUDE_DIR)/xslibrary.hpp
fission_coefficient.o : $(MFEM_HEADER) $(INCLUDE_DIR)/fission_coefficient.hpp $(INCLUDE_DIR)/material_library.hpp $(INCLUDE_DIR)/xslibrary.hpp
