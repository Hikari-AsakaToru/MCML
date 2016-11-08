/*	This file is part of CUDAMCML.

    CUDAMCML is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    CUDAMCML is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with CUDAMCML.  If not, see <http://www.gnu.org/licenses/>.*/
#pragma once
// DEFINES 
#define NUM_BLOCKS_PER_GRID 56 //Keep numblocks a multiple of the #MP's of the GPU (8800GT=14MP)
//The register usage varies with platform. 64-bit Linux and 32.bit Windows XP have been tested.

#ifdef __linux__ //uses 25 registers per thread (64-bit)
	#define NUM_THREADS_PER_BLOCK 320 //Keep above 192 to eliminate global memory access overhead However, keep low to allow enough registers per thread
	#define NUM_THREADS 17920
#endif

#define NUM_THREADS_PER_BLOCK 1024 //Keep above 192 to eliminate global memory access overhead However, keep low to allow enough registers per thread
#define NUM_GRID_PER_BLOCK 20 //Keep above 192 to eliminate global memory access overhead However, keep low to allow enough registers per thread
#define NUM_THREADS_PER_BLOCK_MAKE_RAND 64 //Keep above 192 to eliminate global memory access overhead However, keep low to allow enough registers per thread
#define NUM_DIV_MAKE_RAND 20
#define NUM_THREADS NUM_GRID_PER_BLOCK*NUM_THREADS_PER_BLOCK
#define NON_ACTIVE (255);
#define END_KERNEL (65535);
#define NUM_PHOTON_GPU 100000
#define NFLOATS 5
#define NINTS 5

#include <stdio.h>
#include <math.h>
#include <limits.h>
#include <stdlib.h>
#include <float.h>
#include <string.h>
#include <cstring>
#include <vector>
#include <chrono>




#define NUMSTEPS_GPU 1000
#define PI 3.141592654f
#define RPI 0.318309886f // Pi Rad
#define MAX_LAYERS 100
#define STR_LEN 200

//#define WEIGHT 0.0001f
#define WEIGHTI 429497u //0xFFFFFFFFu*WEIGHT
#define CHANCE 0.1f


// 最大値用
__host__ __device__ struct MaxState{
	int MaxRzSize;
	int MaxRaSize;
};


// TYPEDEFS
__host__ __device__ struct LayerStruct {
	float z_min;		// Layer z_min [cm]
	float z_max;		// Layer z_max [cm]
	float mutr;			// Reciprocal mu_total [cm]
	float mua;			// Absorption coefficient [1/cm]
	float g;			// Anisotropy factor [-]
	float n;			// Refractive index [-]
};

__host__ __device__ struct PhotonStruct{
	float x;		// Global x coordinate [cm]
	float y;		// Global y coordinate [cm]
	float z;		// Global z coordinate [cm]
	float dx;		// (Global, normalized) x-direction
	float dy;		// (Global, normalized) y-direction
	float dz;		// (Global, normalized) z-direction
	unsigned long long weight;			// Photon weight
	int layer;				// Current layer
	unsigned long long Index;
	__device__ __host__ PhotonStruct& PhotonStruct::operator =(const PhotonStruct& b){
		this->dx = b.dx;
		this->dy = b.dy;
		this->dz = b.dz;
		this->x = b.x;
		this->y = b.y;
		this->z = b.z;
		this->layer = b.layer;
		this->weight = b.weight;
	}
};
__host__ __device__ struct PhotonStructAoS{
	float* x;		// Global x coordinate [cm]
	float* y;		// Global y coordinate [cm]
	float* z;		// Global z coordinate [cm]
	float* dx;		// (Global, normalized) x-direction
	float* dy;		// (Global, normalized) y-direction
	float* dz;		// (Global, normalized) z-direction
	unsigned long long* weight;			// Photon weight
	int* layer;				// Current layer
	unsigned long long* Index;
	__device__ __host__ PhotonStructAoS& PhotonStructAoS::operator =(const PhotonStructAoS& b){
		this->dx = b.dx;
		this->dy = b.dy;
		this->dz = b.dz;
		this->x = b.x;
		this->y = b.y;
		this->z = b.z;
		this->layer = b.layer;
		this->weight = b.weight;
	}
};


__host__ __device__ struct DetStruct{
	float dr;		// Detection grid resolution, r-direction [cm]
	float dz;		// Detection grid resolution, z-direction [cm]
	
	unsigned int na;			// Number of grid elements in angular-direction [-]
	unsigned int nr;			// Number of grid elements in r-direction
	unsigned int nz;			// Number of grid elements in z-direction

};
__host__ __device__ struct RecStruct{
	__int64  procTotalTime;		
	__int64  procAvergTime;		
};

__host__ __device__ struct SimulationStruct{
	unsigned long long number_of_photons;
	int ignoreAdetection;
	unsigned int n_layers;
	unsigned long long start_weight;
	char outp_filename[STR_LEN];
	char inp_filename[STR_LEN];
	char AorB;
	unsigned int nDivedSimNum;
	long begin, end;
	DetStruct det;
	LayerStruct* layers;
	unsigned int Seed;
	RecStruct	RecordDoSim;
};


__host__ __device__ struct MemStruct{
	PhotonStruct* p;					// Pointer to structure array containing all the photon data
	unsigned long long* x;				// Pointer to the array containing all the WMC x's
	unsigned int* a;					// Pointer to the array containing all the WMC a's
	unsigned int* thread_active;		// Pointer to the array containing the thread active status
	unsigned int* num_terminated_photons;	//Pointer to a scalar keeping track of the number of terminated photons
	unsigned int* Reserve;
	unsigned long long* Rd_ra;
	unsigned long long* A_rz;			// Pointer to the 2D detection matrix!
	unsigned long long* Tt_ra;
};


#ifdef _NVCC_
template <int ignoreAdetection> __global__ void CalcMCGPU(MemStruct);
__device__ float rand_MWC_oc(unsigned long long*, unsigned int*);
__device__ float rand_MWC_co(unsigned long long*, unsigned int*);
__device__ void LaunchPhoton(PhotonStruct*);
__device__ void Spin(PhotonStruct* p, unsigned long long int* x, unsigned int* a, float g);
__device__ unsigned int Reflect(PhotonStruct*, int, unsigned long long*, unsigned int*);
__device__ unsigned int PhotonSurvive(PhotonStruct*, unsigned long long*, unsigned int*);
__device__ void AtomicAddULL(unsigned long long* address, unsigned int add);
__device__ double SpinTheta(unsigned long long int* x, unsigned int *a, double g);
__device__ void Hop(PhotonStruct* p, float s);
#endif
class cCUDAMCML{
protected:
	cudaDeviceProp  m_sDevProp;
	SimulationStruct* m_simulations;
	MemStruct m_sDeviceMem;
	MemStruct m_sHostMem;
	unsigned int m_nRunCount;
	int m_ProcessTime;
	// 計算するフォトン数
	unsigned long long m_un64NumPhoton;
	// int だと2^32-1 までなので それ以上扱えるように64bitに拡張
	// 確保したメモリ量
	unsigned long long m_un64Membyte;

	// 処理したデータ数
	unsigned long long m_un64PrcsDataNum;
	// dz
	float m_fGridSizeZ;
	float m_fGridSizeR;
	unsigned int	m_unNumGridZ;
	unsigned int	m_unNumGridR;
	unsigned int	m_unNumGridA;
	unsigned int	m_unNumLayer;


public:

	cCUDAMCML();
	~cCUDAMCML();

	void InitGPUStat();
	bool CheckGPU();
	// Mem.cu
	void CopyDeviceToHostMem(MemStruct* HostMem, MemStruct* DeviceMem, SimulationStruct* sim);
	int  CopyHostToDeviceMem(MemStruct* HostMem, MemStruct* DeviceMem, SimulationStruct* sim);
	int DoOneSimulation(SimulationStruct* simulation);
	void RunOldCarnel();
	int MakeRandTableDev();
	int InitDCMem(SimulationStruct* sim);
	int InitPhoton();
	// GPU,CPUメモリの確保
	int InitMallocMem(SimulationStruct* sim);

	int InitMemStructs(MemStruct* HostMem, MemStruct* DeviceMem, SimulationStruct* sim);
	int InitContentsMem(SimulationStruct* sim);
	void FreeMemStructs(MemStruct* HostMem, MemStruct* DeviceMem);

	void FreeSimulationStruct(SimulationStruct* sim, int nRun);
	void FreeFailedSimStrct(SimulationStruct* sim, int nRun);
};
