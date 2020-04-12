#ifndef traverse_eager_h
#define traverse_eager_h
#include "exafmm.h"
#include "xitao.h"
#include <assert.h>
#ifdef USE_XITAO
namespace exafmm {
 class upward_pass: public AssemblyTask { 
  public:  
    Cell* Ci;
    upward_pass(Cell* _ci) : AssemblyTask(1), Ci(_ci){ }
    void execute (int nthread) { 
      int tid = nthread - leader; 
      if(tid==0){ 
        Ci->M.resize(NTERM, 0.0);                                   // Allocate and initialize multipole coefs
        Ci->L.resize(NTERM, 0.0);                                   // Allocate and initialize local coefs
        if(Ci->NCHILD==0) P2M(Ci);                                  // P2M kernel
        M2M(Ci);     
      } 
    }
    void cleanup() { } 
  };
  void build_upward_pass(Cell* Ci, upward_pass* parent) { 
    for (Cell * Cj=Ci->CHILD; Cj!=Ci->CHILD+Ci->NCHILD; Cj++) { // Loop over child cells
      upward_pass* u_pass = new upward_pass(Cj); 
      if(Cj->NCHILD>0)
        build_upward_pass(Cj, u_pass);
      u_pass->make_edge(parent);
      if(Cj->NCHILD==0) gotao_push(u_pass, rand()%gotao_nthreads);
    } 
  } 
   void upwardPass(Cells& cells) {
    Cell* Ci = &cells[0];
    upward_pass* parent = new upward_pass(Ci); 
    upward_pass* child[Ci->NCHILD]; 
    int idx = 0;
    gotao_init();

    for (Cell * Cj=Ci->CHILD; Cj!=Ci->CHILD+Ci->NCHILD; Cj++) { // Loop over child cells
      child[idx] = new upward_pass(Cj); 
      build_upward_pass(Cj, child[idx]);
      child[idx++]->make_edge(parent);
    } 

    gotao_start();
    gotao_fini();
  }

 class P2P_TAO: public AssemblyTask { 
  public:  
    Cell* Ci;
    Cell* Cj;
    P2P_TAO(Cell* _ci, Cell* _cj) : AssemblyTask(1), Ci(_ci), Cj(_cj){ }
    void execute (int nthread) { 
      int tid = nthread - leader; 
      P2P(Ci, Cj, tid, width);
    }
    void cleanup() { } 
  };

  class M2L_TAO: public AssemblyTask { 
  public:  
    Cell* Ci;
    Cell* Cj;
    M2L_TAO(Cell* _ci, Cell* _cj) : AssemblyTask(1), Ci(_ci), Cj(_cj){ }
    void execute (int nthread) { 
      if(nthread == leader) M2L(Ci, Cj);
    }
    void cleanup() { } 
  };
  
  class Sync_TAO: public AssemblyTask { 
  public:  
    Sync_TAO() : AssemblyTask(1){ }
    void execute (int nthread) { }
    void cleanup() { } 
  };
  
  void build_horizontal_pass(Cell* Ci, Cell* Cj, Sync_TAO* parent) { 
   for (int d=0; d<3; d++) dX[d] = Ci->X[d] - Cj->X[d];        // Distance vector from source to target
    real_t R2 = norm(dX) * theta * theta;                       // Scalar distance squared
    if (R2 > (Ci->R + Cj->R) * (Ci->R + Cj->R)) {               // If distance is far enough
      M2L_TAO* tao = new M2L_TAO(Ci, Cj);
      tao->criticality = 1;
      //tao->no_mold = true;
      tao->make_edge(parent);
      gotao_push(tao, rand()%gotao_nthreads);                //  M2L kernel
    } else if (Ci->NCHILD == 0 && Cj->NCHILD == 0) {            // Else if both cells are leafs
      P2P_TAO* tao = new P2P_TAO(Ci, Cj);
      tao->criticality = 1;
      tao->make_edge(parent);
      gotao_push(tao, rand()%gotao_nthreads);                //  P2P kernel
    } else  {
        Sync_TAO* sync_tao = new Sync_TAO();
        sync_tao->no_mold = true;
        sync_tao->criticality = 0;  
        sync_tao->make_edge(parent); 
        if (Cj->NCHILD == 0 || (Ci->R >= Cj->R && Ci->NCHILD != 0)) {// If Cj is leaf or Ci is larger                     
          for (Cell * ci=Ci->CHILD; ci!=Ci->CHILD+Ci->NCHILD; ci++) {// Loop over Ci's children
            build_horizontal_pass(ci, Cj, sync_tao);                              //   Recursive call to target child cells
          }                                                         //  End loop over Ci's children
        } else {                                                    // Else if Ci is leaf or Cj is larger
          for (Cell * cj=Cj->CHILD; cj!=Cj->CHILD+Cj->NCHILD; cj++) {// Loop over Cj's children
            build_horizontal_pass(Ci, cj, sync_tao);                                //   Recursive call to source child cells
          }                                                         //  End loop over Cj's children
        }                                                           // End if for leafs and Ci Cj size
    }
  }


   void horizontalPass(Cells & icells, Cells & jcells) {
     gotao_init();
     Sync_TAO* parent = new Sync_TAO();
     build_horizontal_pass(&icells[0], &jcells[0], parent);
     gotao_start();
     gotao_fini();
   }

}
#endif
namespace exafmm {
#ifndef USE_XITAO
  //! Recursive call to post-order tree traversal for upward pass
  void upwardPass(Cell * Ci) {
    for (Cell * Cj=Ci->CHILD; Cj!=Ci->CHILD+Ci->NCHILD; Cj++) { // Loop over child cells
#pragma omp task untied if(Cj->NBODY > 100)                     //  Start OpenMP task if large enough task
      upwardPass(Cj);                                           //  Recursive call for child cell
    }                                                           // End loop over child cells
#pragma omp taskwait                                            // Synchronize OpenMP tasks
    Ci->M.resize(NTERM, 0.0);                                   // Allocate and initialize multipole coefs
    Ci->L.resize(NTERM, 0.0);                                   // Allocate and initialize local coefs
    if(Ci->NCHILD==0) P2M(Ci);                                  // P2M kernel
    M2M(Ci);                                                    // M2M kernel
  }

  //! Upward pass interface
  void upwardPass(Cells & cells) {
#pragma omp parallel                                            // Start OpenMP
#pragma omp single nowait                                       // Start OpenMP single region with nowait
    upwardPass(&cells[0]);                                      // Pass root cell to recursive call
  }

  //! Recursive call to dual tree traversal for horizontal pass
  void horizontalPass(Cell * Ci, Cell * Cj) {
    for (int d=0; d<3; d++) dX[d] = Ci->X[d] - Cj->X[d];        // Distance vector from source to target
    real_t R2 = norm(dX) * theta * theta;                       // Scalar distance squared
    if (R2 > (Ci->R + Cj->R) * (Ci->R + Cj->R)) {               // If distance is far enough
      M2L(Ci, Cj);                                              //  M2L kernel
    } else if (Ci->NCHILD == 0 && Cj->NCHILD == 0) {            // Else if both cells are leafs
      P2P(Ci, Cj);                                              //  P2P kernel
    } else if (Cj->NCHILD == 0 || (Ci->R >= Cj->R && Ci->NCHILD != 0)) {// If Cj is leaf or Ci is larger
      for (Cell * ci=Ci->CHILD; ci!=Ci->CHILD+Ci->NCHILD; ci++) {// Loop over Ci's children
#pragma omp task untied if(ci->NBODY > 100)                     //   Start OpenMP task if large enough task
        horizontalPass(ci, Cj);                                 //   Recursive call to target child cells
      }                                                         //  End loop over Ci's children
    } else {                                                    // Else if Ci is leaf or Cj is larger
      for (Cell * cj=Cj->CHILD; cj!=Cj->CHILD+Cj->NCHILD; cj++) {// Loop over Cj's children
        horizontalPass(Ci, cj);                                 //   Recursive call to source child cells
      }                                                         //  End loop over Cj's children
    }                                                           // End if for leafs and Ci Cj size
#pragma omp taskwait                                            // Synchronize OpenMP tasks
  }

  //! Horizontal pass interface
  void horizontalPass(Cells & icells, Cells & jcells) {
#pragma omp parallel                                            // Start OpenMP
#pragma omp single nowait                                       // Start OpenMP single region with nowait
    horizontalPass(&icells[0], &jcells[0]);                     // Pass root cell to recursive call
  }
#endif
  //! Recursive call to pre-order tree traversal for downward pass
  void downwardPass(Cell * Cj) {
    L2L(Cj);                                                    // L2L kernel
    if (Cj->NCHILD==0) L2P(Cj);                                 // L2P kernel
    for (Cell * Ci=Cj->CHILD; Ci!=Cj->CHILD+Cj->NCHILD; Ci++) { // Loop over child cells
#pragma omp task untied if(Ci->NBODY > 100)                     //  Start OpenMP task if large enough task
      downwardPass(Ci);                                         //  Recursive call for child cell
    }                                                           // End loop over chlid cells
#pragma omp taskwait                                            // Synchronize OpenMP tasks
  }

  //! Downward pass interface
  void downwardPass(Cells & cells) {
#pragma omp parallel                                            // Start OpenMP
#pragma omp single nowait                                       // Start OpenMP single region with nowait
    downwardPass(&cells[0]);                                    // Pass root cell to recursive call
  }

  //! Direct summation
  void direct(Bodies & bodies, Bodies & jbodies) {
    Cells cells(2);                                             // Define a pair of cells to pass to P2P kernel
    Cell * Ci = &cells[0];                                      // Allocate single target
    Cell * Cj = &cells[1];                                      // Allocate single source
    Ci->BODY = &bodies[0];                                      // Iterator of first target body
    Ci->NBODY = bodies.size();                                  // Number of target bodies
    Cj->BODY = &jbodies[0];                                     // Iterator of first source body
    Cj->NBODY = jbodies.size();                                 // Number of source bodies
    P2P(Ci, Cj);                                                // Evaluate P2P kenrel
  }
}
#endif
