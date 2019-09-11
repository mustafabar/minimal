kernel:
	icpc -I. -Wfatal-errors -fopenmp -mavx -O3 -c -o kernel.o kernel2.cxx
	icpc -Wfatal-errors -fopenmp -mavx -O3 kernel.o -lmpi_cxx
	./a.out -v -n 1000 -c 32 -i 6

serial:
	icpc -I. -Wfatal-errors -fopenmp -mavx -O3 -c -o fmm.o fmm.cxx
	icpc -Wfatal-errors -fopenmp -mavx -O3 fmm.o -lmpi_cxx
	./a.out -v -n 1000 -c 32 -i 6

charmm:
	icpc -I. -Wfatal-errors -fopenmp -mavx -O3 -c -o charmm.o charmm.cxx
	icpc -Wfatal-errors -fopenmp -mavx -O3 charmm.o -lmpi_cxx 
	mpirun -np 1 ./a.out

test_charmm:
	make libcharmm.a
	ifort test_charmm.f90 -fopenmp -L. -lcharmm -lstdc++ -lgomp -lmpi_cxx
	mpirun -n 1 ./a.out ./ water_32.fmm water.out 10

libcharmm.a:
	icpc -I. -Wfatal-errors -fopenmp -mavx -O3 -c -o charmm.o charmm.cxx -DLIBRARY
	ar rv libcharmm.a charmm.o
	ranlib libcharmm.a

clean:
	@find . -name "*.o" -o -name "*.out*" -o -name "*.a" -o -name "*.mod" -o -name "*.dat" -o -name "*.pdb" | xargs rm -rf
