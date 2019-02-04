#include "serialfmm.h"

namespace exafmm {
  class ParallelFMM : public SerialFMM {
  private:
    int EXTERNAL;
    std::ofstream fid;
    std::vector<MPI_Request> requests;

    template<typename T>
    void print(T data) {
      for (int irank=0; irank<MPISIZE; irank++ ) {                // Loop over ranks
	MPI_Barrier(MPI_COMM_WORLD);                              //  Sync processes
	usleep(100);                                              //  Wait 100 milliseconds
	if (MPIRANK == irank) std::cout << data << " ";           //  If it's my turn print "data"
      }                                                           // End loop over ranks
      MPI_Barrier(MPI_COMM_WORLD);                                // Sync processes
      usleep(100);                                                // Wait 100 milliseconds
      if (MPIRANK == MPISIZE-1) std::cout << std::endl;           // New line
    }

    void gatherMultipoles() {
      int i = getGlobKey(IX[gatherLevel],gatherLevel) + globLevelOffset[gatherLevel];
      for_m sendMultipole[0][m] = globMultipole[i][m];
      int numGather = numPartition[gatherLevel][0] * numPartition[gatherLevel][1] * numPartition[gatherLevel][2];
      assert( numGather <= numSendCells ); // resize recvMultipole to avoid this
      int rank;
      MPI_Comm_rank(MPI_COMM_LOCAL,&rank);
      if( rank == 0 ) {
	MPI_Allgather(sendMultipole[0],NTERM,MPI_COMPLEX,
		      recvMultipole[0],NTERM,MPI_COMPLEX,MPI_COMM_GLOBAL);
      }
      MPI_Bcast(recvMultipole[0],numGather*NTERM,MPI_COMPLEX,0,MPI_COMM_LOCAL);
      for( int c=0; c<numGather; c++ ) {
	for_m globMultipole[c+globLevelOffset[gatherLevel]][m] = recvMultipole[c][m];
      }
    }

  public:
    ParallelFMM() {
      int argc(0);
      char **argv;
      MPI_Initialized(&EXTERNAL);
      if(!EXTERNAL) MPI_Init(&argc,&argv);
      MPI_Comm_size(MPI_COMM_WORLD,&MPISIZE);
      MPI_Comm_rank(MPI_COMM_WORLD,&MPIRANK);
      requests.resize(104);
    }
    ~ParallelFMM() {
      if(!EXTERNAL) MPI_Finalize();
    }

    void partitionComm() {
      ivec3 iX;
      for( int i=0; i<MPISIZE; i++ ) sendBodiesCount[i] = 0;
      assert(numBodies % 3 == 0);
      for( int i=0; i<numBodies; i++ ) {
	setGlobIndex(i,iX);
	int sendRank = getGlobKey(iX,maxGlobLevel);
	Rank[i] = sendRank;
	sendBodiesCount[sendRank] += 4;
      }
      for( int i=0; i<MPISIZE; i++ ) assert(sendBodiesCount[i] % 12 == 0);
      MPI_Alltoall(sendBodiesCount,1,MPI_INT,recvBodiesCount,1,MPI_INT,MPI_COMM_WORLD);
      sendBodiesDispl[0] = recvBodiesDispl[0] = 0;
      for( int i=1; i<MPISIZE; i++ ) {
	sendBodiesDispl[i] = sendBodiesDispl[i-1] + sendBodiesCount[i-1];
	recvBodiesDispl[i] = recvBodiesDispl[i-1] + recvBodiesCount[i-1];
      }
      sort(Jbodies,sendJbodies,Index,sendIndex,Rank);
      MPI_Alltoallv(&sendJbodies[0][0], sendBodiesCount, sendBodiesDispl, MPI_FLOAT,
		    &recvJbodies[0][0], recvBodiesCount, recvBodiesDispl, MPI_FLOAT,
		    MPI_COMM_WORLD);
      int newBodies = (recvBodiesDispl[MPISIZE-1] + recvBodiesCount[MPISIZE-1]) / 4;
      for( int i=0; i<newBodies; i++ ) {
	Jbodies[i] = recvJbodies[i];
      }
      sort(Ibodies,sendJbodies,Index,sendIndex,Rank);
      MPI_Alltoallv(&sendJbodies[0][0], sendBodiesCount, sendBodiesDispl, MPI_FLOAT,
		    &recvJbodies[0][0], recvBodiesCount, recvBodiesDispl, MPI_FLOAT,
		    MPI_COMM_WORLD);
      for( int i=0; i<MPISIZE; i++ ) {
	sendBodiesCount[i] /= 4;
	sendBodiesDispl[i] /= 4;
	recvBodiesCount[i] /= 4;
	recvBodiesDispl[i] /= 4;
      }
      MPI_Alltoallv(&sendIndex[0], sendBodiesCount, sendBodiesDispl, MPI_INT,
		    &recvIndex[0], recvBodiesCount, recvBodiesDispl, MPI_INT,
		    MPI_COMM_WORLD);
      numBodies = newBodies;
      for( int i=0; i<numBodies; i++ ) {
	Index[i] = recvIndex[i];
	Ibodies[i] = recvJbodies[i];
      }
    }

    void P2PSend() {
      MPI_Status stats[52];
      int rankOffset = 13 * numLeafs;
      ivec3 iXc;
      getGlobIndex(iXc,MPIRANK,maxGlobLevel);
      ivec3 nunit = numPartition[maxGlobLevel];
      int ileaf = 0;
      int iforward = 0;
      ivec3 iX;
      float commBytes = 0;
      for( iX[2]=-1; iX[2]<=1; iX[2]++ ) {
	for( iX[1]=-1; iX[1]<=1; iX[1]++ ) {
	  for( iX[0]=-1; iX[0]<=1; iX[0]++ ) {
	    if( iX[0] != 0 || iX[1] != 0 || iX[2] != 0 ) {
	      int ibody = bodiesDispl[iforward];
	      int nxmin[3] = {(1 << maxLevel) - DP2P, 0, 0};
	      int nxmax[3] = {1 << maxLevel, 1 << maxLevel, DP2P};
	      ivec3 jX;
	      for( jX[2]=nxmin[iX[2]+1]; jX[2]<nxmax[iX[2]+1]; jX[2]++ ) {
		for( jX[1]=nxmin[iX[1]+1]; jX[1]<nxmax[iX[1]+1]; jX[1]++ ) {
		  for( jX[0]=nxmin[iX[0]+1]; jX[0]<nxmax[iX[0]+1]; jX[0]++, ileaf++ ) {
		    ivec3 jXp = jX;
		    int j = getKey(jXp,maxLevel,false) + rankOffset;
		    sendLeafs[ileaf].begin = ibody;
		    for( int jbody=Leafs[j].begin; jbody<Leafs[j].end; ibody++, jbody++ ) {
		      sendJbodies[ibody] = Jbodies[jbody];
		    }
		    sendLeafs[ileaf].end = ibody;
		  }
		}
	      }
	      if(iforward != 25 ) {
		if( ibody > bodiesDispl[iforward+1] ) std::cout << "ibody: " << ibody << " bodiesDispl: " << bodiesDispl[iforward+1] << " @rank: " << MPIRANK << std::endl;
	      }
	      ivec3 iXp = (iXc - iX + nunit) % nunit;
	      int sendRank = getGlobKey(iXp,maxGlobLevel);
	      iXp = (iXc + iX + nunit) % nunit;
	      int recvRank = getGlobKey(iXp,maxGlobLevel);
	      assert(0<=sendRank && sendRank<MPISIZE);
	      assert(0<=recvRank && recvRank<MPISIZE);
	      int sendDispl = leafsDispl[iforward];
	      int sendCount = leafsCount[iforward];
	      commBytes += sendCount * 2 * 4;
	      MPI_Isend(&sendLeafs[sendDispl].begin,sendCount*2,MPI_INT,
			sendRank,iforward,MPI_COMM_WORLD,&requests[iforward]);
	      int recvDispl = leafsDispl[iforward];
	      int recvCount = leafsCount[iforward];
	      MPI_Irecv(&recvLeafs[recvDispl].begin,recvCount*2,MPI_INT,
			recvRank,iforward,MPI_COMM_WORLD,&requests[iforward+52]);
	      sendDispl = bodiesDispl[iforward];
	      sendCount = bodiesCount[iforward];
	      commBytes += sendCount * 4 * 4;
	      MPI_Isend(&sendJbodies[sendDispl][0],sendCount*4,MPI_FLOAT,
			sendRank,iforward+26,MPI_COMM_WORLD,&requests[iforward+26]);
	      recvDispl = bodiesDispl[iforward];
	      recvCount = bodiesCount[iforward];
	      MPI_Irecv(&recvJbodies[recvDispl][0],recvCount*4,MPI_FLOAT,
			recvRank,iforward+26,MPI_COMM_WORLD,&requests[iforward+78]);
	      iforward++;
	    }
	  }
	}
      }
      MPI_Waitall(52,&requests[0],stats);
    }

    void P2PRecv() {
      MPI_Status stats[52];
      MPI_Waitall(52,&requests[52],stats);
      int ileaf = 0;
      int iforward = 0;
      ivec3 iX;
      for( iX[2]=-1; iX[2]<=1; iX[2]++ ) {
	for( iX[1]=-1; iX[1]<=1; iX[1]++ ) {
	  for( iX[0]=-1; iX[0]<=1; iX[0]++ ) {
	    if( iX[0] != 0 || iX[1] != 0 || iX[2] != 0 ) {
	      assert( ileaf == leafsDispl[iforward] );
	      int rankIndex = (iX[0] + 1) + 3 * (iX[1] + 1) + 9 * (iX[2] + 1);
	      int rankOffset = rankIndex * numLeafs;
	      int ibody = numBodies + bodiesDispl[iforward];
	      int nxmin[3] = {(1 << maxLevel) - DP2P, 0, 0};
	      int nxmax[3] = {1 << maxLevel, 1 << maxLevel, DP2P};
	      ivec3 jX;
	      for( jX[2]=nxmin[iX[2]+1]; jX[2]<nxmax[iX[2]+1]; jX[2]++ ) {
		for( jX[1]=nxmin[iX[1]+1]; jX[1]<nxmax[iX[1]+1]; jX[1]++ ) {
		  for( jX[0]=nxmin[iX[0]+1]; jX[0]<nxmax[iX[0]+1]; jX[0]++, ileaf++ ) {
		    ivec3 jXp = jX;
		    int j = getKey(jXp,maxLevel,false) + rankOffset;
		    Leafs[j].begin = ibody;
		    for( int jbody=recvLeafs[ileaf].begin; jbody<recvLeafs[ileaf].end; ibody++, jbody++ ) {
		      Jbodies[ibody] = recvJbodies[jbody];
		    }
		    Leafs[j].end = ibody;
		  }
		}
	      }
	      iforward++;
	    }
	  }
	}
      }
    }

    void M2LSend(int lev) {
      MPI_Status stats[26];
      int rankOffset = 13 * numCells;
      ivec3 iXc;
      getGlobIndex(iXc,MPIRANK,maxGlobLevel);
      ivec3 nunit = numPartition[maxGlobLevel];
      int nxmin[3] = {(1 << lev) - 2, 0, 0};
      int nxmax[3] = {1 << lev, 1 << lev, 2};
      int i = 0;
      int iforward = 0;
      ivec3 iX;
      float commBytes = 0;
      for( iX[2]=-1; iX[2]<=1; iX[2]++ ) {
	for( iX[1]=-1; iX[1]<=1; iX[1]++ ) {
	  for( iX[0]=-1; iX[0]<=1; iX[0]++ ) {
	    if( iX[0] != 0 || iX[1] != 0 || iX[2] != 0 ) {
	      ivec3 jX;
	      for( jX[2]=nxmin[iX[2]+1]; jX[2]<nxmax[iX[2]+1]; jX[2]++ ) {
		for( jX[1]=nxmin[iX[1]+1]; jX[1]<nxmax[iX[1]+1]; jX[1]++ ) {
		  for( jX[0]=nxmin[iX[0]+1]; jX[0]<nxmax[iX[0]+1]; jX[0]++, i++ ) {
		    ivec3 jXp = jX;
		    int j = getKey(jXp,lev) + rankOffset;
		    for_m sendMultipole[i][m] = Multipole[j][m];
		    commBytes += NTERM * 4;
		  }
		}
	      }
	      ivec3 iXp = (iXc - iX + nunit) % nunit;
	      int sendRank = getGlobKey(iXp,maxGlobLevel);
	      iXp = (iXc + iX + nunit) % nunit;
	      int recvRank = getGlobKey(iXp,maxGlobLevel);
	      int sendDispl = multipoleDispl[lev][iforward];
	      int sendCount = multipoleCount[lev][iforward];
	      MPI_Isend(sendMultipole[sendDispl],sendCount*NTERM,MPI_COMPLEX,
			sendRank,iforward,MPI_COMM_WORLD,&requests[iforward]);
	      int recvDispl = multipoleDispl[lev][iforward];
	      int recvCount = multipoleCount[lev][iforward];
	      MPI_Irecv(recvMultipole[recvDispl],recvCount*NTERM,MPI_COMPLEX,
			recvRank,iforward,MPI_COMM_WORLD,&requests[iforward+26]);
	      iforward++;
	    }
	  }
	}
      }
      MPI_Waitall(26,&requests[0],stats);
    }

    void M2LRecv(int lev) {
      MPI_Status stats[26];
      int nxmin[3] = {(1 << lev) - 2, 0, 0};
      int nxmax[3] = {1 << lev, 1 << lev, 2};
      for( int iforward=0; iforward<26; iforward++ ) {
	int irequest;
	MPI_Waitany(26,&requests[26],&irequest,stats);
	int rankIndex = irequest < 13 ? irequest : irequest+1;
	int iX[3] = {rankIndex % 3, rankIndex / 3 % 3, rankIndex / 9};
	for_3d iX[d]--;
	int i = multipoleDispl[lev][irequest];
	int rankOffset = rankIndex * numCells;
	ivec3 jX;
	for( jX[2]=nxmin[iX[2]+1]; jX[2]<nxmax[iX[2]+1]; jX[2]++ ) {
	  for( jX[1]=nxmin[iX[1]+1]; jX[1]<nxmax[iX[1]+1]; jX[1]++ ) {
	    for( jX[0]=nxmin[iX[0]+1]; jX[0]<nxmax[iX[0]+1]; jX[0]++, i++ ) {
	      ivec3 jXp = jX;
	      int j = getKey(jXp,lev) + rankOffset;
	      for_m Multipole[j][m] = recvMultipole[i][m];
	    }
	  }
	}
      }
    }

    void rootGather() {
#pragma omp parallel for
      for(int i=0;i<numGlobCells;i++){
	globMultipole[i] = complex_t(0);
      }
#pragma omp parallel for
      for( int lev=0; lev<=maxGlobLevel; lev++ ) {
	globLocal[lev] = complex_t(0);
      }
    }

    void globM2MSend(int level) {
      MPI_Status stats[8];
      ivec3 numChild = numPartition[level] / numPartition[level-1];
      ivec3 numStride = numPartition[maxGlobLevel] / numPartition[level];
      ivec3 iX = IX[level];
      ivec3 iXoff = IX[maxGlobLevel] % numStride;
      ivec3 jXoff = (IX[level] / numChild) * numChild;
      int i = getGlobKey(iX,level) + globLevelOffset[level];
      for_m sendMultipole[0][m] = globMultipole[i][m];
      int iforward = 0;
      int numComm = numChild[0] * numChild[1] * numChild[2] - 1;
      float commBytes = 0;
      ivec3 jX;
      for( jX[2]=jXoff[2]; jX[2]<jXoff[2]+numChild[2]; jX[2]++ ) {
	for( jX[1]=jXoff[1]; jX[1]<jXoff[1]+numChild[1]; jX[1]++ ) {
	  for( jX[0]=jXoff[0]; jX[0]<jXoff[0]+numChild[0]; jX[0]++ ) {
	    if( iX[0] != jX[0] || iX[1] != jX[1] || iX[2] != jX[2] ) {
	      ivec3 jXp = iXoff + jX * numStride;
	      int commRank = getGlobKey(jXp,maxGlobLevel);
	      commBytes += NTERM * 4;
	      MPI_Isend(sendMultipole[0],NTERM,MPI_COMPLEX,
			commRank,0,MPI_COMM_WORLD,&requests[iforward]);
	      MPI_Irecv(recvMultipole[iforward],NTERM,MPI_COMPLEX,
			commRank,0,MPI_COMM_WORLD,&requests[iforward+numComm]);
	      iforward++;
	    }
	  }
	}
      }
      MPI_Waitall(numComm,&requests[0],stats);
    }

    void globM2MRecv(int level) {
      MPI_Status stats[8];
      ivec3 numChild;
      for_3d numChild[d] = numPartition[level][d] / numPartition[level-1][d];
      ivec3 iX;
      for_3d iX[d] = IX[level][d];
      ivec3 jXoff;
      for_3d jXoff[d] = (iX[d] / numChild[d]) * numChild[d];
      int iforward = 0;
      int numComm = numChild[0] * numChild[1] * numChild[2] - 1;
      MPI_Waitall(numComm,&requests[numComm],stats);
      ivec3 jX;
      for( jX[2]=jXoff[2]; jX[2]<jXoff[2]+numChild[2]; jX[2]++ ) {
	for( jX[1]=jXoff[1]; jX[1]<jXoff[1]+numChild[1]; jX[1]++ ) {
	  for( jX[0]=jXoff[0]; jX[0]<jXoff[0]+numChild[0]; jX[0]++ ) {
	    if( iX[0] != jX[0] || iX[1] != jX[1] || iX[2] != jX[2] ) {
	      int j = getGlobKey(jX,level) + globLevelOffset[level];
	      for_m globMultipole[j][m] = recvMultipole[iforward][m];
	      iforward++;
	    }
	  }
	}
      }
    }

    void globM2M() {
      int rankOffset = 13 * numCells;
      int i = MPIRANK + globLevelOffset[maxGlobLevel];
      globMultipole[i] = Multipole[rankOffset];
      for( int lev=maxGlobLevel; lev>gatherLevel; lev-- ) {
	start("Comm LET cells");
	globM2MSend(lev);
	globM2MRecv(lev);
	stop("Comm LET cells");
	start("Glob M2M");
	ivec3 numChild = numPartition[lev] / numPartition[lev-1];
	ivec3 jXoff = (IX[lev] / numChild) * numChild;
	int childOffset = globLevelOffset[lev];
	int parentOffset = globLevelOffset[lev-1];
	vec3 diameter;
	for_3d diameter[d] = 2 * RGlob[d] / numPartition[lev][d];
	ivec3 jX;
	for( jX[2]=jXoff[2]; jX[2]<jXoff[2]+numChild[2]; jX[2]++ ) {
	  for( jX[1]=jXoff[1]; jX[1]<jXoff[1]+numChild[1]; jX[1]++ ) {
	    for( jX[0]=jXoff[0]; jX[0]<jXoff[0]+numChild[0]; jX[0]++ ) {
	      ivec3 iX = jX / numChild;
	      int c = getGlobKey(jX,lev) + childOffset;
	      int p = getGlobKey(iX,lev-1) + parentOffset;
	      vec3 dX;
	      for_3d dX[d] = (iX[d] + .5) * numChild[d] * diameter[d] - (jX[d] + .5) * diameter[d];
              M2M(dX,globMultipole[c],globMultipole[p]);
	    }
	  }
	}
	stop("Glob M2M");
      }
      start("Comm LET cells");
      gatherMultipoles();
      stop("Comm LET cells");
      start("Glob M2M");
      for( int lev=gatherLevel; lev>0; lev-- ) {
	ivec3 numChild = numPartition[lev] / numPartition[lev-1];
	int childOffset = globLevelOffset[lev];
	int parentOffset = globLevelOffset[lev-1];
	vec3 diameter;
	for_3d diameter[d] = 2 * RGlob[d] / numPartition[lev][d];
	ivec3 jX;
	for( jX[2]=0; jX[2]<numPartition[lev][2]; jX[2]++ ) {
	  for( jX[1]=0; jX[1]<numPartition[lev][1]; jX[1]++ ) {
	    for( jX[0]=0; jX[0]<numPartition[lev][0]; jX[0]++ ) {
	      ivec3 iX = jX / numChild;
	      int c = getGlobKey(jX,lev) + childOffset;
	      int p = getGlobKey(iX,lev-1) + parentOffset;
	      vec3 dX;
	      for_3d dX[d] = (iX[d] + .5) * numChild[d] * diameter[d] - (jX[d] + .5) * diameter[d];
              M2M(dX,globMultipole[c],globMultipole[p]);
	    }
	  }
	}
      }
      stop("Glob M2M");
    }

    void globM2LSend(int level) {
      MPI_Status stats[26];
      ivec3 numChild = numPartition[level] / numPartition[level-1];
      ivec3 numStride = numPartition[maxGlobLevel] / numPartition[level-1];
      ivec3 iXc = IX[level-1];
      ivec3 iXoff = IX[maxGlobLevel] % numStride;
      int numGroup = numChild[0] * numChild[1] * numChild[2];
      float commBytes = 0;
      int i = 0;
      int iforward = 0;
      ivec3 iX;
      for( iX[2]=-1; iX[2]<=1; iX[2]++ ) {
	for( iX[1]=-1; iX[1]<=1; iX[1]++ ) {
	  for( iX[0]=-1; iX[0]<=1; iX[0]++ ) {
	    if( iX[0] != 0 || iX[1] != 0 || iX[2] != 0 ) {
	      ivec3 jX;
	      for( jX[2]=iXc[2]*numChild[2]; jX[2]<(iXc[2]+1)*numChild[2]; jX[2]++ ) {
		for( jX[1]=iXc[1]*numChild[1]; jX[1]<(iXc[1]+1)*numChild[1]; jX[1]++ ) {
		  for( jX[0]=iXc[0]*numChild[0]; jX[0]<(iXc[0]+1)*numChild[0]; jX[0]++, i++ ) {
		    int j = getGlobKey(jX,level) + globLevelOffset[level];
		    for_m sendMultipole[i][m] = globMultipole[j][m];
		  }
		}
	      }
	      ivec3 iXp = (iXc + iX + numPartition[level-1]) % numPartition[level-1];
	      iXp = iXoff + iXp * numStride;
	      int sendRank = getGlobKey(iXp,maxGlobLevel);
	      commBytes += numGroup * NTERM * 4;
	      MPI_Isend(sendMultipole[iforward*numGroup],numGroup*NTERM,MPI_COMPLEX,
			sendRank,iforward,MPI_COMM_WORLD,&requests[iforward]);
	      iXp = (iXc - iX + numPartition[level-1]) % numPartition[level-1];
	      iXp = iXoff + iXp * numStride;
	      int recvRank = getGlobKey(iXp,maxGlobLevel);
	      MPI_Irecv(recvMultipole[iforward*numGroup],numGroup*NTERM,MPI_COMPLEX,
			recvRank,iforward,MPI_COMM_WORLD,&requests[iforward+26]);
	      iforward++;
	    }
	  }
	}
      }
      MPI_Waitall(26,&requests[0],stats);
    }

    void globM2LRecv(int level) {
      MPI_Status stats[26];
      MPI_Waitall(26,&requests[26],stats);
      ivec3 numChild = numPartition[level] / numPartition[level-1];
      ivec3 iXc = IX[level-1];
      int i = 0;
      int iforward = 0;
      ivec3 iX;
      for( iX[2]=-1; iX[2]<=1; iX[2]++ ) {
	for( iX[1]=-1; iX[1]<=1; iX[1]++ ) {
	  for( iX[0]=-1; iX[0]<=1; iX[0]++ ) {
	    if( iX[0] != 0 || iX[1] != 0 || iX[2] != 0 ) {
	      ivec3 iXp = (iXc - iX + numPartition[level-1]) % numPartition[level-1];
	      ivec3 jX;
	      for( jX[2]=iXp[2]*numChild[2]; jX[2]<(iXp[2]+1)*numChild[2]; jX[2]++ ) {
		for( jX[1]=iXp[1]*numChild[1]; jX[1]<(iXp[1]+1)*numChild[1]; jX[1]++ ) {
		  for( jX[0]=iXp[0]*numChild[0]; jX[0]<(iXp[0]+1)*numChild[0]; jX[0]++, i++ ) {
		    int j = getGlobKey(jX,level) + globLevelOffset[level];
		    for_m globMultipole[j][m] = recvMultipole[i][m];
		  }
		}
	      }
	      iforward++;
	    }
	  }
	}
      }
    }

    void globM2L() {
      for( int lev=maxGlobLevel; lev>0; lev-- ) {
	MPI_Barrier(MPI_COMM_WORLD);
	start("Glob M2L");
	ivec3 nxmin = 0;
	ivec3 nxmax = numPartition[lev-1] - 1;
	ivec3 nunit = numPartition[lev];
	vec3 diameter;
	for_3d diameter[d] = 2 * RGlob[d] / numPartition[lev][d];
	if( numImages != 0 ) {
	  nxmin = -nxmax - 1;
	  nxmax = nxmax * 2 + 1;
	}
	cvecP L = complex_t(0);
	ivec3 iX = IX[lev];
	ivec3 iXp = IX[lev-1];
	ivec3 jXmin =  max(nxmin, iXp - 1)      * numPartition[lev] / numPartition[lev-1];
	ivec3 jXmax = (min(nxmax, iXp + 1) + 1) * numPartition[lev] / numPartition[lev-1];
	ivec3 jX;
	for( jX[2]=jXmin[2]; jX[2]<jXmax[2]; jX[2]++ ) {
	  for( jX[1]=jXmin[1]; jX[1]<jXmax[1]; jX[1]++ ) {
	    for( jX[0]=jXmin[0]; jX[0]<jXmax[0]; jX[0]++ ) {
	      if(jX[0] < iX[0]-1 || iX[0]+1 < jX[0] ||
		 jX[1] < iX[1]-1 || iX[1]+1 < jX[1] ||
		 jX[2] < iX[2]-1 || iX[2]+1 < jX[2]) {
		ivec3 jXp = (jX + nunit) % nunit;
		int j = getGlobKey(jXp,lev) + globLevelOffset[lev];
		vec3 dX;
		for_3d dX[d] = (iX[d] - jX[d]) * diameter[d];
                M2L(dX,globMultipole[j],L);
	      }
	    }
	  }
	}
	globLocal[lev] += L;
	stop("Glob M2L");
      }
    }

    void globL2L() {
      for( int lev=1; lev<=maxGlobLevel; lev++ ) {
	vec3 diameter;
	for_3d diameter[d] = 2 * RGlob[d] / numPartition[lev][d];
	vec3 dX;
	for_3d dX[d] = (IX[lev][d] + .5) * diameter[d] - (IX[lev-1][d] + .5) * 2 * diameter[d];
        L2L(dX,globLocal[lev-1],globLocal[lev]);
      }
      Local[0] += globLocal[maxGlobLevel];
    }
  };
}
