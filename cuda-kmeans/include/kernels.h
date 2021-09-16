#ifndef KERNELS_H
#define KERNELS_H

// DEVICE FUNCTIONS
///////////////////

// assign membership
extern __global__ void find_membership(const float*, const int, const int, float* const, const int, const int,  int*, int*, int*);
extern __global__ void find_membership_global(const float*, const int, const int, float* const, const int, int*, int*, int*);

// update clusters [global memory] [synchronized]
extern __global__ void update_clusters_gmcl(const float*, volatile float*, volatile int*, const int, const int, const int, const int*, const int*,  int*);
extern __global__ void update_clusters_gmdl(const float*, volatile float*, volatile int*, const int, const int, const int, const int*, const int*,  int*);

// update clusters [global memory] [parallelized]
extern __global__ void update_clusters_gmct(const float*, volatile float*, volatile int*, const int, const int, const int, const int*, const int*,  int*);
extern __global__ void update_clusters_gmdt(const float*, volatile float*, volatile int*, const int, const int, const int, const int*, const int*,  int*);

// update clusters [shared memory] [parallelized]
extern __global__ void update_clusters_smct(const float*, volatile float*, volatile int*, const int, const int, const int, const int, const int*, const int*, int*);
extern __global__ void update_clusters_smdt(const float*, volatile float*, volatile int*, const int, const int, const int, const int, const int*, const int*, int*);

// update clusters [shared memory] [synchronized]
extern __global__ void update_clusters_smcl(const float*, volatile float*, volatile int*, const int, const int, const int, const int, const int*, const int*, int*);
extern __global__ void update_clusters_smdl(const float*, volatile float*, volatile int*, const int, const int, const int, const int, const int*, const int*, int*);

// update clusters [global memory] [synchronized]
extern __global__ void update_clusters_global(const float*, float*, int*, const int, const int, const int, const int* __restrict__, const int* __restrict__,  int*);
extern __global__ void update_clusters_global_finegrain(const float*, float*, int*, const int, const int, const int, const int* __restrict__, const int* __restrict__,  int*);

// update clusters [shared cluster, global locks] [synchronized]
extern __global__ void update_clusters_scgcl(const float* __restrict__, volatile float*, volatile int*, const int, const int, const int, const int, const int* __restrict__, const int* __restrict__, int*);
extern __global__ void update_clusters_scgdl(const float* __restrict__, volatile float*, volatile int*, const int, const int, const int, const int, const int* __restrict__, const int* __restrict__, int*);

// UTILITY FUNCTIONS
////////////////////
extern __global__ void normalize_clusters(float*, const int* __restrict__, const int, const int);
extern __global__ void reset_clusters(float*, int*, const int, const int);
extern __global__ void fix_assignments(volatile int*, const int);

// HELPER FUNCTIONS
///////////////////
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
  if (code != cudaSuccess) {
  	printf("YIKES!");
    fprintf(stdout,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
    if (abort) { 
      exit(code);
    }
  }
}
#endif
