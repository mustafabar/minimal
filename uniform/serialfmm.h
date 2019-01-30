#include <mpi.h>
#include "evaluator.h"

namespace exafmm {
  class SerialFMM : public Evaluator {
  protected:
    int bodiesDispl[26];
    int bodiesCount[26];
    int sendBodiesDispl[1024];
    int sendBodiesCount[1024];
    int recvBodiesDispl[1024];
    int recvBodiesCount[1024];
    int multipoleDispl[10][26];
    int multipoleCount[10][26];
    int leafsDispl[26];
    int leafsCount[26];
    ivec3 IX[10];
    int gatherLevel;
    MPI_Comm MPI_COMM_LOCAL, MPI_COMM_GLOBAL;

  private:
    void checkPartition(int *maxPartition) {
      int partitionSize = 1;
      for_3d partitionSize *= maxPartition[d];
      assert( MPISIZE == partitionSize );
      int mpisize = MPISIZE;
      while (mpisize > 0) {
	assert( mpisize % 8 == 0 || mpisize == 1 );
	mpisize /= 8;
      }
      ivec3 checkLevel, partition;
      for_3d partition[d] = maxPartition[d];
      for( int d=0; d<3; d++ ) {
	int lev = 1;
	while( partition[d] != 1 ) {
	  int ndiv = 2;
	  if( (partition[d] % 3) == 0 ) ndiv = 3;
	  partition[d] /= ndiv;
	  lev++;
	}
	checkLevel[d] = lev - 1;
      }
      maxGlobLevel = std::max(std::max(checkLevel[0],checkLevel[1]),checkLevel[2]);
      for_3d numPartition[0][d] = 1;
      for_3d partition[d] = maxPartition[d];
      for( int lev=1; lev<=maxGlobLevel; lev++ ) {
	for( int d=0; d<3; d++ ) {
	  int ndiv = 2;
	  if( (partition[d] % 3) == 0 ) ndiv = 3;
	  if( checkLevel[d] < maxGlobLevel && lev == 1 ) ndiv = 1;
	  numPartition[lev][d] = ndiv * numPartition[lev-1][d];
	  partition[d] /= ndiv;
	}
      }
    }

    void setSendCounts() {
      ivec3 leafsType, bodiesType;
      for_3d leafsType[d] = 1 << (d * maxLevel);
      for_3d bodiesType[d] = leafsType[d] * float(numBodies) / numLeafs * 4;
      int i = 0;
      ivec3 iX;
      bodiesDispl[0] = leafsDispl[0] = 0;
      for( iX[2]=-1; iX[2]<=1; iX[2]++ ) {
	for( iX[1]=-1; iX[1]<=1; iX[1]++ ) {
	  for( iX[0]=-1; iX[0]<=1; iX[0]++ ) {
	    if( iX[0] != 0 || iX[1] != 0 || iX[2] != 0 ) {
	      int zeros = 0;
	      for_3d zeros += iX[d] == 0;
	      bodiesCount[i] = bodiesType[zeros];
	      leafsCount[i] = leafsType[zeros];
	      if( i > 0 ) {
		bodiesDispl[i] = bodiesDispl[i-1] + bodiesCount[i-1];
		leafsDispl[i] = leafsDispl[i-1] + leafsCount[i-1];
	      }
	      i++;
	    }
	  }
	}
      }
      assert( numSendBodies >= bodiesDispl[25] + bodiesCount[25] );
      assert( bodiesDispl[25] + bodiesCount[25] > 0 );
      assert( numSendLeafs == leafsDispl[25] + leafsCount[25] );
      int sumSendCells = 0;
      for( int lev=1; lev<=maxLevel; lev++ ) {
	int multipoleType[3] = {8, 4*(1<<lev), 2*(1<<(2*lev))};
	multipoleDispl[lev][0] = 0;
	i = 0;
	for( iX[2]=-1; iX[2]<=1; iX[2]++ ) {
	  for( iX[1]=-1; iX[1]<=1; iX[1]++ ) {
	    for( iX[0]=-1; iX[0]<=1; iX[0]++ ) {
	      if( iX[0] != 0 || iX[1] != 0 || iX[2] != 0 ) {
		int zeros = 0;
		for_3d zeros += iX[d] == 0;
		multipoleCount[lev][i] = multipoleType[zeros];
		sumSendCells += multipoleCount[lev][i];
		if( i > 0 ) {
		  multipoleDispl[lev][i] = multipoleDispl[lev][i-1] + multipoleCount[lev][i-1];
		}
		i++;
	      }
	    }
	  }
	}
      }
      assert( numSendCells == sumSendCells );
    }

  protected:
    inline void getIndex(int i, int *iX, real_t diameter) const {
#if NOWRAP
      i = (i / 3) * 3;
#endif
      for_3d iX[d] = int((Jbodies[i][d] + R0 - X0[d]) / diameter);
    }

    inline void setGlobIndex(int i, int *iX) const {
#if NOWRAP
      i = (i / 3) * 3;
#endif
      for_3d iX[d] = int(Jbodies[i][d] / (2 * R0));
      for_3d iX[d] = iX[d] % numPartition[maxGlobLevel][d];
    }

    void sort(vec4 *bodies, vec4 *buffer, int *index, int *ibuffer, int *key) const {
      int Imax = key[0];
      int Imin = key[0];
      for( int i=0; i<numBodies; i++ ) {
	Imax = std::max(Imax,key[i]);
	Imin = std::min(Imin,key[i]);
      }
      int numBucket = Imax - Imin + 1;
      std::vector<int> bucket(numBucket);
      for( int i=0; i<numBucket; i++ ) bucket[i] = 0;
      for( int i=0; i<numBodies; i++ ) bucket[key[i]-Imin]++;
      for( int i=1; i<numBucket; i++ ) bucket[i] += bucket[i-1];
      for( int i=numBodies-1; i>=0; --i ) {
	bucket[key[i]-Imin]--;
	int inew = bucket[key[i]-Imin];
	ibuffer[inew] = index[i];
	for_4d buffer[inew][d] = bodies[i][d];
      }
    }

  public:
    void allocate(int N, int L, int Im) {
      maxLevel = L;
      numBodies = N;
      numImages = Im;
      numCells = ((1 << 3 * (L + 1)) - 1) / 7;
      numLeafs = 1 << 3 * L;
      numSendCells = 64 * L + 48 * ((1 << (L + 1)) - 2) + 12 * (((1 << (2 * L + 2)) - 1) / 3 - 1);
      numSendLeafs = 8 + 12 * (1 << L) + 6 * (1 << (2 * L));
      numSendBodies = numSendLeafs * float(numBodies) / numLeafs * 4;
      float memory = 0;
      memory += numBodies * 4 * sizeof(real_t);
      memory += (numBodies + numSendBodies) * 4 * sizeof(real_t);
      memory += 27 * numCells * MTERM * sizeof(real_t);
      memory += numCells * LTERM * sizeof(real_t);
      memory += 27 * numLeafs * 2 * sizeof(int);
      memory += 2 * MPISIZE * MTERM * sizeof(real_t);
      memory += 10 * LTERM * sizeof(real_t);
      memory += numSendBodies * 4 * sizeof(float);
      memory += numSendBodies * 4 * sizeof(float);
      memory += numSendCells * MTERM * sizeof(float);
      memory += numSendCells * MTERM * sizeof(float);
      memory += numSendLeafs * 2 * sizeof(int);
      memory += numSendLeafs * 2 * sizeof(int);
      //std::cout << "Memory: " << memory/1e6 << " MB" << std::endl;
      Index = new int [2*numBodies];
      Rank = new int [2*numBodies];
      sendIndex = new int [2*numBodies];
      recvIndex = new int [2*numBodies];
      Leafs = new int [27*numLeafs][2]();
      sendLeafs = new int [numSendLeafs][2]();
      recvLeafs = new int [numSendLeafs][2]();
      Ibodies = new vec4 [2*numBodies]();
      Jbodies = new vec4 [2*numBodies+numSendBodies];
      Multipole = new real_t [27*numCells][MTERM];
      Local = new real_t [numCells][LTERM]();
      globMultipole = new real_t [2*MPISIZE][MTERM]();
      globLocal = new real_t [10][LTERM]();
      sendJbodies = new vec4 [2*numBodies+numSendBodies];
      recvJbodies = new vec4 [2*numBodies+numSendBodies];
      sendMultipole = new float [numSendCells][MTERM]();
      recvMultipole = new float [numSendCells][MTERM]();
    }

    void deallocate() {
      delete[] Index;
      delete[] Rank;
      delete[] sendIndex;
      delete[] recvIndex;
      delete[] Leafs;
      delete[] sendLeafs;
      delete[] recvLeafs;
      delete[] Ibodies;
      delete[] Jbodies;
      delete[] Multipole;
      delete[] Local;
      delete[] globMultipole;
      delete[] globLocal;
      delete[] sendJbodies;
      delete[] recvJbodies;
      delete[] sendMultipole;
      delete[] recvMultipole;
    }

    void partitioner(int level) {
      int mpisize = MPISIZE;
      ivec3 maxPartition = 1;
      int dim = 0;
      while( mpisize != 1 ) {
	int ndiv = 2;
	if( (mpisize % 3) == 0 ) ndiv = 3;
	maxPartition[dim] *= ndiv;
	mpisize /= ndiv;
	dim = (dim + 1) % 3;
      }
      checkPartition(maxPartition);
      numGlobCells = 0;
      for( int lev=0; lev<=maxGlobLevel; lev++ ) {
	globLevelOffset[lev] = numGlobCells;
	numGlobCells += numPartition[lev][0] * numPartition[lev][1] * numPartition[lev][2];
      }
      getGlobIndex(IX[maxGlobLevel],MPIRANK,maxGlobLevel);
      for( int lev=maxGlobLevel; lev>0; lev-- ) {
	for_3d IX[lev-1][d] = IX[lev][d] * numPartition[lev-1][d] / numPartition[lev][d];
      }
      setSendCounts();
      gatherLevel = level;
      if(gatherLevel > maxGlobLevel) gatherLevel = maxGlobLevel;
#if EXAFMM_SERIAL
#else
      ivec3 iX, numChild;
      for_3d numChild[d] = numPartition[maxGlobLevel][d] / numPartition[gatherLevel][d];
      for_3d iX[d] = IX[maxGlobLevel][d] % numChild[d];
      int key = iX[0] + (iX[1] + iX[2] * numChild[1]) * numChild[0];
      int color = getGlobKey(IX[gatherLevel],gatherLevel);
      MPI_Comm_split(MPI_COMM_WORLD, color, key, &MPI_COMM_LOCAL);
      MPI_Comm_split(MPI_COMM_WORLD, key, color, &MPI_COMM_GLOBAL);
#endif
    }

    void sortBodies() const {
      int *key = new int [numBodies];
      real_t diameter = 2 * R0 / (1 << maxLevel);
      ivec3 iX = 0;
      for( int i=0; i<numBodies; i++ ) {
	getIndex(i,iX,diameter);
	key[i] = getKey(iX,maxLevel);
      }
      sort(Jbodies,sendJbodies,Index,sendIndex,key);
      for( int i=0; i<numBodies; i++ ) {
	Index[i] = sendIndex[i];
	for_4d Jbodies[i][d] = sendJbodies[i][d];
      }
      delete[] key;
    }

    void buildTree() const {
      int rankOffset = 13 * numLeafs;
      for( int i=rankOffset; i<numLeafs+rankOffset; i++ ) {
	Leafs[i][0] = Leafs[i][1] = 0;
      }
      real_t diameter = 2 * R0 / (1 << maxLevel);
      ivec3 iX = 0;
      getIndex(0,iX,diameter);
      int ileaf = getKey(iX,maxLevel,false) + rankOffset;
      Leafs[ileaf][0] = 0;
      for( int i=0; i<numBodies; i++ ) {
	getIndex(i,iX,diameter);
	int inew = getKey(iX,maxLevel,false) + rankOffset;
	if( ileaf != inew ) {
	  Leafs[ileaf][1] = Leafs[inew][0] = i;
	  ileaf = inew;
	}
      }
      Leafs[ileaf][1] = numBodies;
      for( int i=rankOffset; i<numLeafs+rankOffset; i++ ) {
	//assert( Leafs[i][1] != Leafs[i][0] );
      }
    }

    void periodicM2L() {
      real_t M[MTERM];
#if EXAFMM_SERIAL
      for_m M[m] = Multipole[13*numCells][m];
#else
      for_m M[m] = globMultipole[0][m];
#endif
      real_t L[LTERM];
      for_l L[l] = 0;
      for( int lev=1; lev<numImages; lev++ ) {
	vec3 diameter;
	for_3d diameter[d] = 2 * RGlob[d] * std::pow(3.,lev-1);
	ivec3 jX;
	for( jX[2]=-4; jX[2]<=4; jX[2]++ ) {
	  for( jX[1]=-4; jX[1]<=4; jX[1]++ ) {
	    for( jX[0]=-4; jX[0]<=4; jX[0]++ ) {
	      if(jX[0] < -1 || 1 < jX[0] ||
		 jX[1] < -1 || 1 < jX[1] ||
		 jX[2] < -1 || 1 < jX[2]) {
		vec3 dX;
		for_3d dX[d] = jX[d] * diameter[d];
		real_t invR2 = 1. / (dX[0] * dX[0] + dX[1] * dX[1] + dX[2] * dX[2]);
		real_t invR  = sqrt(invR2);
		real_t C[LTERM];
		getCoef(C,dX,invR2,invR);
		M2LSum(L,C,M);
	      }
	    }
	  }
	}
	real_t M3[MTERM];
	for_m M3[m] = 0;
	ivec3 iX;
	for( iX[2]=-1; iX[2]<=1; iX[2]++ ) {
	  for( iX[1]=-1; iX[1]<=1; iX[1]++ ) {
	    for( iX[0]=-1; iX[0]<=1; iX[0]++ ) {
	      vec3 dX;
	      for_3d dX[d] = iX[d] * diameter[d];
	      real_t C[LTERM];
	      C[0] = 1;
	      powerM(C,dX);
	      for_m M3[m] += C[m] * M[0];
	      M2MSum(M3,C,M);
	    }
	  }
	}
	for_m M[m] = M3[m];
      }
#if EXAFMM_SERIAL
      for_l Local[0][l] += L[l];
#else
      for_l globLocal[0][l] += L[l];
#endif
    }
  };
}
