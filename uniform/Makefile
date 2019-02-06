serial:
	mpiicpc -I. -DEXAFMM_SERIAL -Wfatal-errors -fopenmp -mavx -O3 -c -o fmm.o fmm.cxx
	mpiicpc -Wfatal-errors -fopenmp -mavx -O3 fmm.o -lmpi_cxx
	./a.out -v -n 1000 -c 32 -i 6

parallel:
	mpiicpc -I. -Wfatal-errors -fopenmp -mavx -O3 -c -o fmm_mpi.o fmm.cxx
	mpiicpc -Wfatal-errors -fopenmp -mavx -O3 fmm_mpi.o -lmpi_cxx 
	mpirun -np 8 ./a.out -v -n 1000 -c 32 -i 6

charmm:
	mpiicpc -I. -Wfatal-errors -fopenmp -mavx -O3 -c -o charmm.o charmm.cxx
	mpiicpc -Wfatal-errors -fopenmp -mavx -O3 charmm.o -lmpi_cxx 
	mpirun -np 8 ./a.out -v -n 1000 -c 32 -i 6

clean:
	@find . -name "*.o" -o -name "*.out*" -o -name "*.dat" | xargs rm -rf
