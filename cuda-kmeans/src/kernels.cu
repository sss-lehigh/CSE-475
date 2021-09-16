#include <stdio.h>
#include <cfloat>
#include "kernels.h"

//// HELPER KERNELS FOR KMEANS
//////////////////////////////
/*
 * calculates the distance between a datapoint and the provided cluster
 * IN: dataset, clusters
 * OUT: membership 
 * Note: Based on the distance function used in STAMP
 */
__forceinline__ __device__ float calc_distance(
		const float* example, int nfeatures, float* cluster) {
	float dist = 0.0;
	float corr = 0.0;
	for (int i = 0; i < nfeatures; ++i) {
		float y = __fmaf_rd(example[i] - cluster[i], example[i] - cluster[i], corr);
		float t = dist + y;
		corr = y - (t - dist);
		dist = t;
	}
	return sqrtf(dist);
}

/*
 * normalizes the clusters after all thread blocks finish updating centroids
 */
__global__ void normalize_clusters(float* clusters,
		const int* __restrict__ nmembers, const int nclusters,
		const int nfeatures) {
	// find the centroid by getting average
	int p_idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (p_idx < nclusters * nfeatures) {
		int count = nmembers[p_idx / nfeatures];
		if (count < 1) {
			count = 1;
		}
		clusters[p_idx] = clusters[p_idx] / count;
	}
}

/*
 * Reset centroids for new centroid calculation
 */
__global__ void reset_clusters(float* clusters, int* nmembers,
		const int nclusters, const int nfeatures) {
	int p_idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (p_idx < nclusters * nfeatures) {
		clusters[p_idx] = 0.0;
		nmembers[p_idx / nfeatures] = 0;
	}
}
//////////////////////
//////////////////////

//// ASSIGN MEMBERSHIP
//////////////////////
// finds memberhsip for each datapoint using shared memory by copying in clusters and writing clusters 
__global__ void find_membership(const float* data,
		const int npoints, const int nfeatures, float* const clusters,
		const int nclusters, const int cchunk, int* assignments,
		int* assignments_prev, int* update) {
	extern __shared__ float s_clusters[];
	int p_idx = blockIdx.x * blockDim.x + threadIdx.x;

	// in the following
	// cluster_offset = where to start copying clusters
	// nclusters = total number of clusters
	// cchunk = how many clusters can fit in shared memory at once
	float min_dist = FLT_MAX;
	int ass = -1;
	const float limit = 0.99999;
	for (int cluster_offset = 0; cluster_offset < nclusters; cluster_offset +=
			cchunk) {
		// copy in cluster segment 
		if (threadIdx.x < cchunk && cluster_offset + threadIdx.x < nclusters) {
			for (int i = 0; i < nfeatures; ++i) {
				s_clusters[(threadIdx.x * nfeatures) + i] = clusters[(cluster_offset
						+ threadIdx.x) * nfeatures + i];
			}
		}
		__syncthreads();

		// find if new closest cluster
		if (p_idx < npoints) {
			// find assignments for the clusters in shared memory
			for (int i = 0; i < cchunk && cluster_offset + i < nclusters; ++i) {
				//printf("target cluster: %d\n", cluster_offset + i);
				float dist = calc_distance(&data[p_idx * nfeatures], nfeatures,
						&s_clusters[i * nfeatures]);
				if ((dist / min_dist) < limit) {
					min_dist = dist;
					ass = i;
				}
			}
		}
		__syncthreads();
	}

	// if assignment changed then update
	if (p_idx < npoints) {
		if (ass != -1 && ass != assignments[p_idx]) {
			++(*update);
			assignments_prev[p_idx] = assignments[p_idx];
			assignments[p_idx] = ass;
		}
	}
}

// finds membership for each datapoint using global memory only
__global__ void find_membership_global(const float* data,
		const int npoints, const int nfeatures, float* const clusters,
		const int nclusters, int* assignments, int* assignments_prev, int* update) {
	int p_idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (p_idx >= npoints) {
		return;
	}

	float min_dist = FLT_MAX;
	int ass = -1;
	const float limit = 0.99999;
	for (int i = 0; i < nclusters; ++i) {
		float dist = calc_distance(&data[p_idx * nfeatures], nfeatures,
				&clusters[i * nfeatures]);
		if ((dist / min_dist) < limit) {
			min_dist = dist;
			ass = i;
		}
	}

	if (ass != -1 && ass != assignments[p_idx]) {
		++(*update);
		assignments[p_idx] = ass;
	}
	else {
		assignments_prev[p_idx] = assignments[p_idx];
	}
}
/////////////////////////////////
/////////////////////////////////

//// UPDATE CLUSTERS [no locking]
/////////////////////////////////
__global__ void update_clusters_gmct(const float* data,
		volatile float* clusters, volatile int* nmembers, const int nclusters,
		const int npoints, const int nfeatures,
		const int* assignments,
		const int* assignments_prev, int* locks) {
	extern __shared__ float s_mem[];
	int p_idx = blockDim.x * blockIdx.x + threadIdx.x; // global target cluster
	if (p_idx >= nclusters) {
		return;
	} // if not a real cluster, than exit

	// update centroids
	for (int i = 0; i < npoints; ++i) {
		int ass = assignments[i];
		if (ass == p_idx) {
			for (int j = 0; j < nfeatures; ++j) {
				clusters[p_idx * nfeatures + j] += data[i * nfeatures + j];
			}
			++nmembers[p_idx];
		}
	}
}

__global__ void update_clusters_gmdt(const float* data,
		volatile float* clusters, volatile int* nmembers, const int nclusters,
		const int npoints, const int nfeatures,
		const int* assignments,
		const int* assignments_prev, int* locks) {
	extern __shared__ float s_mem[];
	int p_idx = blockDim.x * blockIdx.x + threadIdx.x; // global target cluster
	if (p_idx >= nclusters * nfeatures) {
		return;
	} // if not a cluster, than exit

	// update centroids
	int dim_offset = p_idx % nfeatures;
	int t_cluster = (int)(p_idx / nfeatures);
	for (int i = 0; i < npoints; ++i) {
		int ass = assignments[i];
		if (ass == t_cluster) {
			clusters[p_idx] += data[i * nfeatures + dim_offset];
			if (dim_offset == 0) {
				++nmembers[t_cluster];
			}
		}
	}
}

/*
 * SM-CT
 */
// Simple shared memory implementation, using one thread for each cluster (mimicks Source{d} implementation, maintains STAMP update approach) [cluster-centric]
__global__ void update_clusters_smct(const float* data,
		volatile float* clusters, volatile int* nmembers, const int nclusters,
		const int cchunk, const int npoints, const int nfeatures,
		const int* assignments,
		const int* assignments_prev, int* locks) {
	extern __shared__ float s_mem[];
	int p_idx = blockDim.x * blockIdx.x + threadIdx.x; // global target cluster
	if (threadIdx.x > cchunk || p_idx >= nclusters) {
		return;
	} // if not a real cluster, than exit

	float* s_clusters = reinterpret_cast<float*>(s_mem);
	int* s_nmembers = reinterpret_cast<int*>(s_clusters + (cchunk * nfeatures));

	int cluster_offset = blockIdx.x * cchunk;
	if (threadIdx.x < cchunk) {
		for (int i = 0; i < nfeatures; ++i) {
			s_clusters[threadIdx.x * nfeatures + i] = 0.0;
		}
		s_nmembers[threadIdx.x] = 0;
	}
	__syncthreads(); // needed to ensure sharedmem init is finished...

	// update centroids
	for (int i = 0; i < npoints; ++i) {
		if (p_idx >= cluster_offset && p_idx < cluster_offset + cchunk) {
			int ass = assignments[i];
			if (ass == p_idx) {
				int s_target = ass % cchunk;
				for (int j = 0; j < nfeatures; ++j) {
					s_clusters[s_target * nfeatures + j] += data[i * nfeatures + j];
				}
				++s_nmembers[s_target];
			}
		}
	}
	__syncthreads();

	// copy out
	if (p_idx >= cluster_offset && p_idx < cluster_offset + cchunk) {
		int s_target = p_idx % cchunk;
		for (int i = 0; i < nfeatures; ++i) {
			clusters[p_idx * nfeatures + i] += s_clusters[s_target * nfeatures + i];
		}
		nmembers[p_idx] = s_nmembers[s_target];
	}
}

__global__ void update_clusters_smdt(const float* data,
		volatile float* clusters, volatile int* nmembers, const int nclusters,
		const int cchunk, const int npoints, const int nfeatures,
		const int* assignments,
		const int* assignments_prev, int* locks) {
	extern __shared__ float s_mem[];
	int p_idx = blockDim.x * blockIdx.x + threadIdx.x; // global target cluster and dimension
	if (p_idx >= nclusters * nfeatures) {
		return;
	} // if not a real cluster, than exit

	float* s_clusters = reinterpret_cast<float*>(s_mem);
	int* s_nmembers = reinterpret_cast<int*>(s_clusters + (cchunk * nfeatures));

	int cluster_offset = blockIdx.x * cchunk;
	if (threadIdx.x < cchunk) {
		for (int i = 0; i < nfeatures; ++i) {
			s_clusters[threadIdx.x * nfeatures + i] = 0.0;
		}
		s_nmembers[threadIdx.x] = 0;
	}
	__syncthreads(); // needed to ensure sharedmem init is finished...

	// update centroids
	int t_cluster = p_idx / nfeatures;
	int dim_offset = p_idx % nfeatures;
	for (int i = 0; i < npoints; ++i) {
		if (t_cluster >= cluster_offset && t_cluster < cluster_offset + cchunk) {
			int ass = assignments[i];
			if (ass == t_cluster) {
				int s_target = t_cluster % cchunk;
				s_clusters[s_target * nfeatures + dim_offset] += data[i * nfeatures + dim_offset];
				if (p_idx % nfeatures == 0) {
					++s_nmembers[s_target];
				}
			}
		}
	}
	__syncthreads();

	// copy out
	if (t_cluster >= cluster_offset && t_cluster < cluster_offset + cchunk) {
		int s_target = t_cluster % cchunk;
		clusters[p_idx] += s_clusters[s_target * nfeatures + dim_offset];
		if (p_idx % nfeatures == 0) {
			nmembers[t_cluster] = s_nmembers[s_target];
		}
	}
}


// threadblock-per-centroid implementation
// Each thread block is assigned a cluster, its threads are then assigned to a feature
// No locking is needed, because there is no overlap between threads (even from different blocks)
// One thread in each tb is responsible for incrementing number of members found for its centroid
// Data is global (can be optimized to be stored read-only)
__global__ void update_clusters_shared_tb(const float* data,
		volatile float* clusters, volatile int* nmembers, const int nclusters,
		const int npoints, const int nfeatures, const int* assignments,
		const int* assignments_prev) {
	extern __shared__ float s_mem[];
	__shared__ int s_nmembers;
	float* s_cluster = reinterpret_cast<float*>(s_mem);

	// ignore unused threads
	if (threadIdx.x >= nfeatures) {
		return;
	}

	// init cluster
	s_cluster[threadIdx.x] = 0.0;
	if (threadIdx.x == 0) {
		s_nmembers = 0;
	}

	// go through all data and update cluster accordingly
	for (int i = 0; i < npoints; ++i) {
		if (assignments[i] == blockIdx.x) {
			s_cluster[threadIdx.x] += data[i * nfeatures + threadIdx.x];
			if (threadIdx.x == 0) {
				++s_nmembers;
			}
		}
	}

	// copy cluster to global memory
	clusters[blockIdx.x * nfeatures + threadIdx.x] = s_cluster[threadIdx.x];
	if (threadIdx.x == 0) {
		nmembers[blockIdx.x] = s_nmembers;
	}
}
/////////////////////////////////
