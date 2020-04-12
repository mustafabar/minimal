XITAO_HOME=/home/musabdu/Repos/xitao/
include ${XITAO_HOME}/makefile.sched

CXX = g++ 
CXXFLAGS+= -DUSE_XITAO
CXXFLAGS+= -g -O3 -Wall -Wfatal-errors -fopenmp -I${XITAO_HOME}/include 
LDFLAGS = -L${XITAO_HOME}/lib -lxitao

all:
	@make kernel
	@make fmm

kernel: kernel.cxx
	$(CXX) $(CXXFLAGS) $? -o $@
	./kernel 10
	./kernel 20
	./kernel 30

fmm: fmm.cxx
	$(CXX) $(CXXFLAGS) $? -o $@ $(LDFLAGS) -DEXAFMM_EAGER
	./fmm
	$(CXX) $(CXXFLAGS) $? -o $@ $(LDFLAGS) -DEXAFMM_LAZY
	./fmm

clean:
	$(RM) ./*.o ./kernel ./fmm
