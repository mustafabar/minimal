include ../Makefile.am.include

if EXAFMM_HAVE_MPI
bin_PROGRAMS = fmm fmm_mpi fmm_ijhpca_mpi
fmm_SOURCES = fmm.cxx
fmm_CPPFLAGS = $(AM_CPPFLAGS) -DEXAFMM_SERIAL
fmm_mpi_SOURCES = fmm.cxx
fmm_mpi_CPPFLAGS = $(AM_CPPFLAGS)
fmm_ijhpca_mpi_SOURCES = fmm.cxx
fmm_ijhpca_mpi_CPPFLAGS = $(AM_CPPFLAGS) -DEXAFMM_IJHPCA

run_fmm: fmm
	./fmm -v -n 1000 -c 32 -i 3
run_fmm_mpi: fmm_mpi
	mpirun -np 8 ./fmm_mpi -v -n 1000 -c 32 -i 3
run_fmm_ijhpca_mpi: fmm_ijhpca_mpi
	mpirun -np 8 ./fmm_ijhpca_mpi -v -n 62500 -c 32 -i 1
endif
