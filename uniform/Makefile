serial:
	mpiicpc -I. -DEXAFMM_SERIAL -Wfatal-errors -fopenmp -mavx -O3 -c -o fmm.o fmm.cxx
	mpiicpc -Wfatal-errors -fopenmp -mavx -O3 -o fmm.out fmm.o -lmpi_cxx
	./fmm.out -v -n 1000 -c 32 -i 3

parallel:
	mpiicpc -I. -Wfatal-errors -fopenmp -mavx -O3 -c -o fmm_mpi.o fmm.cxx
	mpiicpc -Wfatal-errors -fopenmp -mavx -O3 -o fmm_mpi.out fmm_mpi.o -lmpi_cxx 
	mpirun -np 8 ./fmm_mpi.out -v -n 1000 -c 32 -i 3

clean:
	@find . -name "*.o" -o -name "*.out*" -o -name "*.dat" | xargs rm -rf
