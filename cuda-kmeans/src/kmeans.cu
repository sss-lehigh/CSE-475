#include <fstream>
#include <string>
#include <cstring>
#include <iostream>
#include <sstream>
#include <limits>
#include <chrono>
#include <time.h>
#include "kernels.h"
#include "kmeans.h"

#define M_THREADS 1024
#define U_THREADS 1024
#define R_THREADS 1024
#define MAX_SHMEM 49000

#define GM_CT 0
#define GM_DT 1
#define SM_CT 2
#define SM_DT 3

void setFindMembershipParameters(data_t*, int, int&, int&, int&, int&);
void setUpdateParameters(int, data_t*, int, int&, int&, int&, int&);
void setResetParameters(data_t*, int, int&, int&);
int getNumBlocksNeeded(int, int);
std::string getAlgoString(int);

int cuda_kmeans(int algo, int k, data_t* d, int threads) {
        // init clusters...
	float* h_data;
	float* h_clusters;
	float* h_distances;
	int* h_assignments;
	int* h_assignments_prev;
	int* h_nmembers;
	int* h_locks;
	int* h_fg_locks;

	float* data;
	float* clusters;
	float* distances;
	int* assignments;
	int* assignments_prev;
	int* nmembers;
	int* locks;
	int* fg_locks;

	gpuErrchk(cudaMallocHost(&h_data, sizeof(float) * d->numPoints * d->numAttrs));
	gpuErrchk(cudaMallocHost(&h_clusters, sizeof(float) * k * d->numAttrs));
	gpuErrchk(cudaMallocHost(&h_distances, sizeof(float) * d->numPoints));
	gpuErrchk(cudaMallocHost(&h_assignments, sizeof(int) * d->numPoints));
	gpuErrchk(cudaMallocHost(&h_assignments_prev, sizeof(int) * d->numPoints));
	gpuErrchk(cudaMallocHost(&h_nmembers, sizeof(int) * k));
	gpuErrchk(cudaMallocHost(&h_locks, sizeof(int) * k));
	gpuErrchk(cudaMallocHost(&h_fg_locks, sizeof(int) * k * d->numAttrs));

	gpuErrchk(cudaMalloc(&data, sizeof(float) * d->numPoints * d->numAttrs));
	gpuErrchk(cudaMalloc(&clusters, sizeof(float) * k * d->numAttrs));
	gpuErrchk(cudaMalloc(&distances, sizeof(float) * d->numPoints));
	gpuErrchk(cudaMalloc(&assignments, sizeof(int) * d->numPoints));
	gpuErrchk(cudaMalloc(&assignments_prev, sizeof(int) * d->numPoints));
	gpuErrchk(cudaMalloc(&nmembers, sizeof(int) * k));
	gpuErrchk(cudaMalloc(&locks, sizeof(int) * k));
	gpuErrchk(cudaMalloc(&fg_locks, sizeof(int) * k * d->numAttrs));

//  gpuErrchk(cudaMallocManaged(&data, sizeof(float) * d->numPoints * d->numAttrs));
//  gpuErrchk(cudaMallocManaged(&clusters, sizeof(float) * k * d->numAttrs));
//  gpuErrchk(cudaMallocManaged(&distances, sizeof(float) * d->numPoints));
//  gpuErrchk(cudaMallocManaged(&assignments, sizeof(int) * d->numPoints));
//  gpuErrchk(cudaMallocManaged(&assignments_prev, sizeof(int) * d->numPoints));
//  gpuErrchk(cudaMallocManaged(&nmembers, sizeof(int) * k));
//  gpuErrchk(cudaMallocManaged(&locks, sizeof(int) * k));

	for (int i = 0; i < d->numPoints; ++i) {
		for (int j = 0; j < d->numAttrs; ++j) {
			h_data[i * d->numAttrs + j] = d->data[i][j];
		}
	}

	int rand_idx[k];
	setRandomIndices(rand_idx, k);

	for (int i = 0; i < k; ++i) {
		for (int j = 0; j < d->numAttrs; ++j) {
			h_clusters[i * d->numAttrs + j] = d->data[rand_idx[i]][j];
			h_fg_locks[i] = 0;
		}
		h_locks[i] = 0;
		h_nmembers[i] = 0;
	}

	for (int i = 0; i < d->numPoints; ++i) {
		h_distances[i] = std::numeric_limits<float>::max();
		h_assignments[i] = -1;
		h_assignments_prev[i] = -1;
	}

	gpuErrchk(
			cudaMemcpyAsync(data, h_data, sizeof(float) * d->numPoints * d->numAttrs,
					cudaMemcpyHostToDevice));
	gpuErrchk(
			cudaMemcpyAsync(clusters, h_clusters, sizeof(float) * k * d->numAttrs,
					cudaMemcpyHostToDevice));
	gpuErrchk(
			cudaMemcpyAsync(distances, h_distances, sizeof(float) * d->numPoints,
					cudaMemcpyHostToDevice));
	gpuErrchk(
			cudaMemcpyAsync(assignments, h_assignments, sizeof(int) * d->numPoints,
					cudaMemcpyHostToDevice));
	gpuErrchk(
			cudaMemcpyAsync(assignments_prev, h_assignments_prev,
					sizeof(int) * d->numPoints, cudaMemcpyHostToDevice));
	gpuErrchk(
			cudaMemcpyAsync(nmembers, h_nmembers, sizeof(int) * k,
					cudaMemcpyHostToDevice));
	gpuErrchk(
			cudaMemcpyAsync(locks, h_locks, sizeof(int) * k, cudaMemcpyHostToDevice));
	gpuErrchk(
			cudaMemcpyAsync(fg_locks, h_fg_locks, sizeof(int) * k * d->numAttrs, cudaMemcpyHostToDevice));
	gpuErrchk(cudaDeviceSynchronize());

	// setup threads, blocks and shared memory for find membership kernel
	int m_threads = 0, m_blocks = 0, m_chunk = 0, m_sharedmem = 0;
	setFindMembershipParameters(d, k, m_threads, m_blocks, m_chunk, m_sharedmem);

	// setup threads, blocks and shared memory for the update kernel
	int u_threads = 0, u_blocks = 0, u_chunk = 0, u_sharedmem = 0;
	u_threads = threads;
	setUpdateParameters(algo, d, k, u_threads, u_blocks, u_chunk, u_sharedmem);
	//printf("%d\n", u_chunk);
	//if(u_chunk != 0) printf("%d\n", (k / u_chunk) + 1);

	// setup threads and blocks for the cluster reset kernel
	int r_threads = 0, r_blocks = 0;
	setResetParameters(d, k, r_threads, r_blocks);

	// loop variables
	int loop = 0;
	bool done = false;
	int *update;
	gpuErrchk(cudaMallocManaged(&update, sizeof(int))); // accessible on both host and device

	// timing setup
	uint64_t avg_runtime = 0;
	std::chrono::duration<double, std::milli> total_runtime;

	// K-means
	do {
		auto g_start = std::chrono::high_resolution_clock::now();
		*update = 0;

		// assign each datapoint to a cluster
                // find_membership<<<m_blocks, m_threads, m_sharedmem>>>(data, d->numPoints,
                //                 d->numAttrs, clusters, k, m_chunk, assignments, assignments_prev,
                //                 update);
		find_membership_global<<<m_blocks, m_threads>>>(data, d->numPoints,
				d->numAttrs, clusters, k, assignments, assignments_prev,
				update);
		gpuErrchk(cudaDeviceSynchronize());

                int reassignments = 0;
                uint64_t iter_runtime = 0;
		gpuErrchk(
				cudaMemcpyAsync(h_assignments, assignments, sizeof(int) * d->numPoints,
						cudaMemcpyDeviceToHost));
		gpuErrchk(
				cudaMemcpyAsync(h_assignments_prev, assignments_prev,
						sizeof(int) * d->numPoints, cudaMemcpyDeviceToHost));

		gpuErrchk(cudaDeviceSynchronize());
		for(int i = 0; i < d->numPoints; ++i) {
			 if(h_assignments[i] != h_assignments_prev[i]) ++reassignments;
		}

		// check if any were reassigned
		done = true;
		if (*update > 0) {
			done = false;
		}

		// if reassigned, then recalculate clusters
		if (!done) {
			// reset clusters
			reset_clusters<<<r_blocks, r_threads>>>(clusters, nmembers, k,
					d->numAttrs);
			gpuErrchk(cudaDeviceSynchronize());

			// constant pointers for kernels
			const float* const_data = data;
			const int* const_ass = assignments;
			const int* const_ass_prev = assignments_prev;
			volatile float* vol_clusters = clusters;
			volatile int* vol_nmembers = nmembers;

			struct timespec cstart, cend;
			// cluster update
			switch (algo) {
			case GM_CT:
                                clock_gettime(CLOCK_REALTIME, &cstart);
				update_clusters_gmct<<<u_blocks, u_threads, u_sharedmem>>>(const_data, vol_clusters,
						vol_nmembers, k, d->numPoints, d->numAttrs, const_ass, const_ass_prev,
						locks);
				gpuErrchk(cudaDeviceSynchronize())
			        normalize_clusters<<<r_blocks, r_threads>>>(clusters, nmembers, k, d->numAttrs);
				gpuErrchk(cudaDeviceSynchronize())
                                clock_gettime(CLOCK_REALTIME, &cend);
				break;
			case GM_DT:
                                clock_gettime(CLOCK_REALTIME, &cstart);
				update_clusters_gmdt<<<u_blocks, u_threads, u_sharedmem>>>(const_data, vol_clusters,
						vol_nmembers, k, d->numPoints, d->numAttrs, const_ass, const_ass_prev,
						locks);
				gpuErrchk(cudaDeviceSynchronize())
			  normalize_clusters<<<r_blocks, r_threads>>>(clusters, nmembers, k, d->numAttrs);
				gpuErrchk(cudaDeviceSynchronize())
                                clock_gettime(CLOCK_REALTIME, &cend);
				break;
			case SM_CT:
                                clock_gettime(CLOCK_REALTIME, &cstart);
				update_clusters_smct<<<u_blocks, u_threads, u_sharedmem>>>(
						const_data, clusters, nmembers, k, u_chunk, d->numPoints, d->numAttrs, const_ass,
						const_ass_prev, locks);
				gpuErrchk(cudaDeviceSynchronize())
			        normalize_clusters<<<r_blocks, r_threads>>>(clusters, nmembers, k, d->numAttrs);
				gpuErrchk(cudaDeviceSynchronize())
                                clock_gettime(CLOCK_REALTIME, &cend);
				break;
			case SM_DT:
                                clock_gettime(CLOCK_REALTIME, &cstart);
				update_clusters_smdt<<<u_blocks, u_threads, u_sharedmem>>>(
						const_data, clusters, nmembers, k, u_chunk, d->numPoints, d->numAttrs, const_ass,
						const_ass_prev, locks);
				gpuErrchk(cudaDeviceSynchronize())
			  normalize_clusters<<<r_blocks, r_threads>>>(clusters, nmembers, k, d->numAttrs);
				gpuErrchk(cudaDeviceSynchronize())
                                clock_gettime(CLOCK_REALTIME, &cend);
				break;
			}

			// normalize clusters
			gpuErrchk(cudaDeviceSynchronize());

			// calculate run times
			auto g_stop = std::chrono::high_resolution_clock::now();
                        iter_runtime = (cend.tv_sec - cstart.tv_sec) * 1000000000 + (cend.tv_nsec - cstart.tv_nsec);
			avg_runtime += iter_runtime; 
			total_runtime += g_stop - g_start;
		}

		// start sanity check
		int total_assigned = 0;
		gpuErrchk(cudaMemcpy(h_nmembers, nmembers, sizeof(int) * k, cudaMemcpyDeviceToHost));
		gpuErrchk(cudaDeviceSynchronize())
		for (int i = 0; i < k; ++i) {
			total_assigned += h_nmembers[i];
		}
		if (total_assigned != d->numPoints) {
			std::cout << "[FAILED]" << std::endl;
			std::cout << "-- loop: " << loop << std::endl;
			std::cout << "-- total assigned: " << total_assigned << std::endl;
			std::cout << "-- number of points: " << d->numPoints << std::endl;
			exit(1);
		}

                //printf("%d\t%lu\n", reassignments, iter_runtime);
	} while (!done && ++loop < 500);

	float tot_rt = total_runtime.count();
	long double updt_rt = ((long double)(avg_runtime) / loop) / 1000000;

//	gpuErrchk(
//			cudaMemcpy(h_clusters, clusters, sizeof(float) * k * d->numAttrs,
//					cudaMemcpyDeviceToHost))
	std::string algoString = getAlgoString(algo);
	std::printf("%s\t%d\t%f\t%LF\t%d\t%d\t%d\t%d\n", algoString.c_str(), k, tot_rt, updt_rt,
			loop, u_threads, u_blocks, u_sharedmem);
	return 1;
}

void setFindMembershipParameters(data_t* d, int k, int& m_threads, int& m_blocks, int& m_chunk, int& m_sharedmem) {
	m_threads = M_THREADS;
	m_blocks = (
			d->numPoints % m_threads == 0 ?
					d->numPoints / m_threads : d->numPoints / m_threads + 1);
	m_chunk = MAX_SHMEM / (sizeof(float) * d->numAttrs);
	m_sharedmem = sizeof(float) * (d->numAttrs * m_chunk);
}

void setUpdateParameters(int algo, data_t* d, int k, int& u_threads, int& u_blocks, int& u_chunk, int& u_sharedmem){
	int need;
	switch (algo) {
	case GM_CT: // GM-CT
		need = k;
		u_blocks = getNumBlocksNeeded(need, u_threads);
		break;
	case GM_DT: // GM-DT
		need = k * d->numAttrs;
		u_blocks = getNumBlocksNeeded(need, u_threads);
		break;
	case SM_CT: // SM-CT
		need = k;
		u_chunk = MAX_SHMEM / (sizeof(float) * d->numAttrs + sizeof(int));
		u_chunk = (u_chunk > u_threads ? u_threads : u_chunk);
		u_threads = u_chunk;
		u_blocks = getNumBlocksNeeded(need, u_threads);
		u_sharedmem = sizeof(float) * (d->numAttrs * u_chunk) + sizeof(int) * u_chunk;
		break;
	case SM_DT: // SM-DT
		need = k * d->numAttrs;
		u_chunk = MAX_SHMEM / (sizeof(float) * d->numAttrs + sizeof(int));
		u_chunk = (u_chunk > u_threads / d->numAttrs ? u_threads / d->numAttrs : u_chunk);
		u_threads = u_chunk * d->numAttrs;
		u_blocks = getNumBlocksNeeded(need, u_threads);
		u_sharedmem = sizeof(float) * (d->numAttrs * u_chunk) + sizeof(int) * u_chunk;
		break;
    default:
		std::cout << "Invalid algorithm number provided... exiting." << std::endl;
		exit(1);
	}
}

void setResetParameters(data_t* d, int k, int& r_threads, int& r_blocks) {
	r_threads = R_THREADS;
	r_blocks = getNumBlocksNeeded(k * d->numAttrs, r_threads);
}

int getNumBlocksNeeded(int need, int threads_per_block) {
	return (need % threads_per_block == 0 ? need / threads_per_block : need / threads_per_block + 1);
}

std::string getAlgoString(int algo) {
	switch(algo){
		case GM_CT: return "GM-CT";
		case GM_DT: return "GM-DT";
		case SM_CT: return "SM-CT";
		case SM_DT: return "SM-DT";
		default: return "";
	}
}
