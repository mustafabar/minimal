kernel:
	mpiicpc -I. -DEXAFMM_SERIAL -Wfatal-errors -fopenmp -mavx -O3 -c -o kernel.o kernel.cxx
	mpiicpc -Wfatal-errors -fopenmp -mavx -O3 kernel.o -lmpi_cxx
	./a.out -v -n 1000 -c 32 -i 6

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
	mpirun -np 1 ./a.out

test_charmm:
	make clean libcharmm.a
	mpiifort test_charmm.f90 -fopenmp -L. -lcharmm -lstdc++ -lgomp -lmpi_cxx
	mpirun -n 1 ./a.out ./ water_32.fmm water.out 0
#	mpirun -n 1 ./a.out

libcharmm.a:
	mpiicpc -I. -Wfatal-errors -fopenmp -mavx -O3 -c -o charmm.o charmm.cxx -DLIBRARY
	ar ruv libcharmm.a charmm.o
	ranlib libcharmm.a

clean:
	@find . -name "*.o" -o -name "*.out*" -o -name "*.a" -o -name "*.mod" -o -name "*.dat" -o -name "*.pdb" | xargs rm -rf
