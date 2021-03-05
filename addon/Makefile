# Makefile

include ../Makefile.include

CPP_FILES = bilininteg.cpp \
			lininteg.cpp

HPP_FILES = bilininteg.hpp \
			lininteg.hpp \
			tensor.hpp

OBJ = $(CPP_FILES:.cpp=.o)

default : $(OBJ)

%.o : %.cpp
	$(MPICXX) $(MPICXXFLAGS) $(INCLUDE) -c $<

clean :
	$(RM) $(OBJ)

bilininteg.o : $(MFEM_HEADER) bilininteg.hpp tensor.hpp
lininteg.o : $(MFEM_HEADER) lininteg.hpp tensor.hpp
