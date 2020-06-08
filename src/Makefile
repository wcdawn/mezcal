# Makefile

CXX = clang++
CXXFLAGS = -Wall -pedantic-errors -std=c++17

MPICXX = mpicxx

LD = nvcc
LDFLAGS = --expt-extended-lambda -arch=sm_60 -ccbin $(MPICXX)

MFEM_DIR = ../build/mfem
MFEM_INCLUDE = $(MFEM_DIR)
MFEM_HEADER = $(MFEM_INCLUDE)/mfem.hpp
MFEM_LIB = $(MFEM_DIR)/libmfem.a

HYPRE_DIR = ../hypre/src/hypre
HYPRE_INCLUDE = $(HYPRE_DIR)/include
HYPRE_LIB = $(HYPRE_DIR)/lib/libHYPRE.a

METIS_DIR = ../metis-5.1.0
METIS_INCLUDE = $(METIS_DIR)/include
METIS_LIB = $(METIS_DIR)/build/Linux-x86_64/libmetis/libmetis.a

ifeq ($(METHOD), dbg)
CXXFLAGS += -O0 -g
else
CXXFLAGS += -O3
endif

CPP_FILES = main.cpp analytic.cpp

HPP_FILES = 

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

$(EXEC) : $(OBJ)
	OMPI_CXX=$(CXX) $(LD) $(LDFLAGS) $^ -o $@ $(MFEM_LIB) $(HYPRE_LIB) $(METIS_LIB)

%.o : %.cpp
	$(CXX) $(CXXFLAGS) -I$(MFEM_INCLUDE) -I$(HYPRE_INCLUDE) -I$(METIS_INCLUDE) -c $<

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

main.o : analytic.o $(MFEM_HEADER)
analytic.o : $(MFEM_HEADER) analytic.hpp
