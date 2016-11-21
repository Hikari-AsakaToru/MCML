#include "cuda_runtime.h"

#define _NVCC_
#include "CUDAMCML_GPGPU.h"
#include <curand_kernel.h>

#define _ERR_GPU_SIM_RND_ 1
#define _ERR_GPU_SIM_MEMCPY_ 2
#define _ERR_GPU_SIM_LOOP_ 3
#define _ERR_GPU_SIM_LANCH_PHOTON_ 4
#define _ERR_GPU_SIM_MCML_ 5
#define _ERR_GPU_SIM_ANOTHER_ 0xFF
#define _SUCCESS_GPU_SIM_ 0
#define PARTIALREFLECTION 1     
#define GNUCC 0
/* 1=split photon, 0=statistical reflection. */

// MemStruct m_sDeviceMem;

__device__ __constant__ unsigned int num_photons_dc[1];
__device__ __constant__ unsigned int n_layers_dc[1];
__device__ __constant__ unsigned long long start_weight_dc[1];
__device__ __constant__ LayerStruct layers_dc[MAX_LAYERS];
__device__ __constant__ DetStruct det_dc[1];
__device__ __constant__ unsigned int dc_Seed[1];
__device__ unsigned int nInitRngLoop=0;

__shared__ PhotonStruct dsh_sPhoton[NUM_THREADS_PER_BLOCK];


//
// MCML計算の本体
// 
template <int ignoreAdetection> __global__ void MCd(MemStruct DeviceMem)
{
	//Block index
	int bx = blockIdx.x;

	//Thread index
	int tx = threadIdx.x;


	//First element processed by the block
	int begin = blockDim.x*bx;



	unsigned long long int x = DeviceMem.x[begin + tx];//coherent
	unsigned int a = DeviceMem.a[begin + tx];//coherent

	float s;	//step length
	//p->s = s;   //PhotonStructのsとの同期が必要？

	unsigned long long index, w, index_old,DataPos;
	index_old = 0;
	w = 0;
	unsigned int w_temp;
	DataPos = *DeviceMem.num_terminated_photons;

	PhotonStruct p = DeviceMem.p[begin + tx];
	

	int new_layer;

	//First, make sure the thread (photon) is active
	unsigned int ii = 0;
	if (!DeviceMem.thread_active[begin + tx]){
		ii = NUMSTEPS_GPU;
	}

	for (; ii<NUMSTEPS_GPU; ii++) //this is the main while loop
	{
		if (layers_dc[p.layer].mutr != FLT_MAX)
			p.s = -__logf(rand_MWC_oc(&x, &a))*layers_dc[p.layer].mutr;//sample step length [cm] //HERE AN OPEN_OPEN FUNCTION WOULD BE APPRECIATED
		else
			p.s = 100.0f;//temporary, say the step in glass is 100 cm.

		//Check for layer transitions and in case, calculate s
		new_layer = p.layer;
		if (p.z + s*p.dz<layers_dc[p.layer].z_min){ new_layer--; s = __fdividef(layers_dc[p.layer].z_min - p.z, p.dz); } //Check for upwards reflection/transmission & calculate new s
		if (p.z + s*p.dz>layers_dc[p.layer].z_max){ new_layer++; s = __fdividef(layers_dc[p.layer].z_max - p.z, p.dz); } //Check for downward reflection/transmission

		p.x += p.dx*s;
		p.y += p.dy*s;
		p.z += p.dz*s;

		if (p.z>layers_dc[p.layer].z_max)p.z = layers_dc[p.layer].z_max;//needed?
		if (p.z<layers_dc[p.layer].z_min)p.z = layers_dc[p.layer].z_min;//needed?

		if (new_layer != p.layer)
		{
			// set the remaining step length to 0
			s = 0.0f;

			if (Reflect(&p, new_layer, &x, &a) == 0u)//Check for reflection
			{ // Photon is transmitted
				if (new_layer == 0)
				{ //Diffuse reflectance
					index = __float2int_rz(acosf(-p.dz)*2.0f*RPI*det_dc[0].na)*det_dc[0].nr + min(__float2int_rz(__fdividef(sqrtf(p.x*p.x + p.y*p.y), det_dc[0].dr)), (int)det_dc[0].nr - 1);
					AtomicAddULL(&DeviceMem.Rd_ra[index], p.weight);
					p.weight = 0; // Set the remaining weight to 0, effectively killing the photon
				}
				if (new_layer > *n_layers_dc)
				{	//Transmitted
					index = __float2int_rz(acosf(p.dz)*2.0f*RPI*det_dc[0].na)*det_dc[0].nr + min(__float2int_rz(__fdividef(sqrtf(p.x*p.x + p.y*p.y), det_dc[0].dr)), (int)det_dc[0].nr - 1);
					AtomicAddULL(&DeviceMem.Tt_ra[index], p.weight);
					p.weight = 0; // Set the remaining weight to 0, effectively killing the photon
				}
			}
		}

		//w=0;

		if (s > 0.0f)
		{
			// Drop weight (apparently only when the photon is scattered)
			w_temp = __float2uint_rn(layers_dc[p.layer].mua*layers_dc[p.layer].mutr*__uint2float_rn(p.weight));
			p.weight -= w_temp;


			//w = __float2uint_rn(layers_dc[p.layer].mua*layers_dc[p.layer].mutr*__uint2float_rn(p.weight));
			//p.weight -= w;

			if (ignoreAdetection == 0) // Evaluated at compiletime!
			{
				index = (min(__float2int_rz(__fdividef(p.z, det_dc[0].dz)), (int)det_dc[0].nz - 1)*det_dc[0].nr + min(__float2int_rz(__fdividef(sqrtf(p.x*p.x + p.y*p.y), det_dc[0].dr)), (int)det_dc[0].nr - 1));
				if (index == index_old)
				{
					w += w_temp;
					//p.weight -= __float2uint_rn(layers_dc[p.layer].mua*layers_dc[p.layer].mutr*__uint2float_rn(p.weight)); 
				}
				else// if(w!=0)
				{
					AtomicAddULL(&DeviceMem.A_rz[index_old], w);
					index_old = index;
					w = w_temp;
				}

			}

			Spin(&p, &x, &a, layers_dc[p.layer].g);
		}




		if (!PhotonSurvive(&p, &x, &a)) // Check if photons survives or not
		{
			if (atomicAdd(DeviceMem.num_terminated_photons, 1u) < (*num_photons_dc))
			{	// Ok to launch another photon
				LaunchPhoton(&p);//Launch a new photon
			}
			else
			{	// No more photons should be launched. 
				DeviceMem.thread_active[DataPos] = 0u; // Set thread to inactive
				ii = NUMSTEPS_GPU;				// Exit main loop
			}

		}
	}//end main for loop!
	if (ignoreAdetection == 1 && w != 0)
		AtomicAddULL(&DeviceMem.A_rz[index_old], w);

	__syncthreads();//necessary?

	//save the state of the MC simulation in global memory before exiting
	DeviceMem.p[DataPos] = p;	//This one is incoherent!!!
	DeviceMem.x[DataPos] = x; //this one also seems to be coherent


}//end MCd
template <int ignoreAdetection> __global__ void CalcMCGPU(MemStruct DeviceMem)
{
	//Block index
	int bx = blockIdx.x;

	//Thread index
	int tx = threadIdx.x;


	//First element processed by the block
	int begin = blockDim.x*bx;
	if (DeviceMem.thread_active[begin + tx] == 65535){
		return;
	}
	if (DeviceMem.thread_active[begin + tx]){
		auto temp = atomicAdd(DeviceMem.num_terminated_photons, 1ul);
		if (temp > *num_photons_dc){
			DeviceMem.thread_active[begin + tx] = 65535;
			return;
		}
		DeviceMem.thread_active[begin + tx] = 0;
	}

	DeviceMem.thread_active[begin + tx] = 0;
	unsigned long long int x = DeviceMem.x[begin + tx];	//coherent
	unsigned int a = DeviceMem.a[begin + tx];			//coherent
	dsh_sPhoton[tx] = DeviceMem.p[begin + tx];
	float s; //step length
	

	unsigned int index, index_old;
	index_old = 0;
	unsigned long long w,w_temp;
	w = 0;

	int new_layer;

	//First, make sure the thread (photon) is active
	unsigned int ii = 0;


	for (; ii<NUMSTEPS_GPU; ii++) //this is the main while loop
	{
		// Rand Make
		// 桁あふれ出ないか確認
		if (layers_dc[dsh_sPhoton[tx].layer].mutr != FLT_MAX){
			// 乱数による距離生成
			s = -__logf(rand_MWC_oc(&x, &a))*layers_dc[dsh_sPhoton[tx].layer].mutr;	//sample step length [cm] //HERE AN OPEN_OPEN FUNCTION WOULD BE APPRECIATED
		}else{
			// 一時的に100 cm代入
			s = 100.0f;															//temporary, say the step in glass is 100 cm.
		}
		// Hop_Drop() mcml_go
		//Check for layer transitions and in case, calculate s
		new_layer = dsh_sPhoton[tx].layer;
		// 現在のレイヤーよりも上に移動してるかを確認
		if (dsh_sPhoton[tx].z + s*dsh_sPhoton[tx].dz<layers_dc[dsh_sPhoton[tx].layer].z_min){
			new_layer--; 
			s = __fdividef(layers_dc[dsh_sPhoton[tx].layer].z_min - dsh_sPhoton[tx].z, dsh_sPhoton[tx].dz); 
		} //Check for upwards reflection/transmission & calculate new s
		// 現在のレイヤーよりも下に移動してるかを確認
		if (dsh_sPhoton[tx].z + s*dsh_sPhoton[tx].dz>layers_dc[dsh_sPhoton[tx].layer].z_max){
			new_layer++;
			s = __fdividef(layers_dc[dsh_sPhoton[tx].layer].z_max - dsh_sPhoton[tx].z, dsh_sPhoton[tx].dz); 
		} //Check for downward reflection/transmission

		// 位置を代入
		dsh_sPhoton[tx].x += dsh_sPhoton[tx].dx*s;
		dsh_sPhoton[tx].y += dsh_sPhoton[tx].dy*s;
		dsh_sPhoton[tx].z += dsh_sPhoton[tx].dz*s;
//		Hop(&dsh_sPhoton[tx],s);
		if (dsh_sPhoton[tx].z > layers_dc[dsh_sPhoton[tx].layer].z_max){
			dsh_sPhoton[tx].z = layers_dc[dsh_sPhoton[tx].layer].z_max;//needed?
		}
		if (dsh_sPhoton[tx].z < layers_dc[dsh_sPhoton[tx].layer].z_min){
			dsh_sPhoton[tx].z = layers_dc[dsh_sPhoton[tx].layer].z_min;//needed?
		}
		//　レイヤー変化していた場合

		if (new_layer != dsh_sPhoton[tx].layer)
		{
			// set the remaining step length to 0
			s = 0.0f;

			// 反射するか確認
			if (Reflect(&dsh_sPhoton[tx], new_layer, &x, &a) == 0u)//Check for reflection
			{ 
				// Photon is transmitted　光子が伝達
				if (new_layer == 0)
				{	// Diffuse reflectance　拡散反射
					// __float2int_rz ・・・float  => int　への型変換(小数点切り捨て？)
//					index = __float2int_rz(acosf(-dsh_sPhoton[tx].dz)*2.0f*RPI*det_dc[0].na)*det_dc[0].nr + min(__float2int_rz(__fdividef(sqrtf(dsh_sPhoton[tx].x*dsh_sPhoton[tx].x + dsh_sPhoton[tx].y*dsh_sPhoton[tx].y), det_dc[0].dr)), (int)det_dc[0].nr - 1);
					index = __float2int_rz(__fdividef(acosf(-dsh_sPhoton[tx].dz) , (PI / det_dc[0].na)))*det_dc[0].nr + min(__float2int_rz(__fdividef(__fsqrt_rz(dsh_sPhoton[tx].x*dsh_sPhoton[tx].x + dsh_sPhoton[tx].y*dsh_sPhoton[tx].y), det_dc[0].dr)), (int)det_dc[0].nr - 1);
					AtomicAddULL(&DeviceMem.Rd_ra[index], dsh_sPhoton[tx].weight);
					dsh_sPhoton[tx].weight = 0;
//					RecordR(dsh_sPhoton[tx]->rr, DeviceMem.In_Ptr, &dsh_sPhoton[tx], DeviceMem.Out_Ptr);//rをどうやって持ってくればいいのか
					RemodelRecordR(DeviceMem, &dsh_sPhoton[tx]);//rをどうやって持ってくればいいのか
				}
				if (new_layer > *n_layers_dc)
				{	//Transmitted　透過
					index = __float2int_rz(__fdividef(acosf(dsh_sPhoton[tx].dz), (PI / det_dc[0].na)))*det_dc[0].nr + min(__float2int_rz(__fdividef(__fsqrt_rz(dsh_sPhoton[tx].x*dsh_sPhoton[tx].x + dsh_sPhoton[tx].y*dsh_sPhoton[tx].y), det_dc[0].dr)), (int)det_dc[0].nr - 1);
					AtomicAddULL(&DeviceMem.Tt_ra[index], dsh_sPhoton[tx].weight);
					dsh_sPhoton[tx].weight = 0;
				}
			}
		}
		//w=0;
		if (s > 0.0f)
		{
			// Drop weight (apparently only when the photon is scattered) 光子の質量減少
			w_temp = __float2uint_rn(layers_dc[dsh_sPhoton[tx].layer].mua*layers_dc[dsh_sPhoton[tx].layer].mutr*__uint2float_rn(dsh_sPhoton[tx].weight));
			dsh_sPhoton[tx].weight -= w_temp;

			if (ignoreAdetection == 0) // Evaluated at compiletime!
			{
				index = (min(__float2int_rz(__fdividef(dsh_sPhoton[tx].z, det_dc[0].dz)), (int)det_dc[0].nz - 1)*det_dc[0].nr + min(__float2int_rz(__fdividef(sqrtf(dsh_sPhoton[tx].x*dsh_sPhoton[tx].x + dsh_sPhoton[tx].y*dsh_sPhoton[tx].y), det_dc[0].dr)), (int)det_dc[0].nr - 1));
				if (index == index_old)
				{
					w += w_temp;
					//sharedp.weight -= __float2uint_rn(layers_dc[sharedp.layer].mua*layers_dc[sharedp.layer].mutr*__uint2float_rn(sharedp.weight)); 
				}
				else// if(w!=0)
				{
					AtomicAddULL(&DeviceMem.A_rz[index_old], w);
					index_old = index;
					w = w_temp;
				}

			}
			// 角度計算
			Spin(&dsh_sPhoton[tx], &x, &a, layers_dc[dsh_sPhoton[tx].layer].g);
		}



		if (!PhotonSurvive(&dsh_sPhoton[tx], &x, &a)) // Check if photons survives or not
		{
			DeviceMem.thread_active[begin + tx] = 1;
			LaunchPhoton(&dsh_sPhoton[tx]);
			break;
		}
	}//end main for loop!
	if (ignoreAdetection == 1 && w != 0)
		AtomicAddULL(&DeviceMem.A_rz[index_old], w);

	__syncthreads();//necessary?

	DeviceMem.x[begin + tx] = x; //this one also seems to be coherent
	DeviceMem.p[begin + tx] = dsh_sPhoton[tx]; //this one also seems to be coherent


}//end MCd

__device__  void LaunchPhoton(LayerStruct  * Layerspecs_Ptr,
	PhotonStruct* p,
	OutStruct    * Out_Ptr,
	InputStruct  * In_Ptr)
{
	// We are currently not using the RNG but might do later
	//float input_fibre_radius = 0.03;//[cm]
	//p->x=input_fibre_radius*sqrtf(rand_MWC_co(x,a));

	p->dead = 0;
	p->layer = 1;
	p->s = 0;
	p->sleft = 0;

	p->x = 0.0f;
	p->y = 0.0f;
	p->z = 0.0f;
	p->dx = 0.0f;
	p->dy = 0.0f;
	p->dz = 1.0f;

	p->layer = 1;
	p->weight = *start_weight_dc; //specular reflection!

}
__device__  void LaunchPhoton(PhotonStruct* p)
{

	p->dead = 0;
	p->layer = 1;
	p->s = 0;
	p->sleft = 0;

	p->x = 0.0f;
	p->y = 0.0f;
	p->z = 0.0f;
	p->dx = 0.0f;
	p->dy = 0.0f;
	p->dz = 1.0f;

	p->layer = 1;
	p->weight = *start_weight_dc; //specular reflection!

}

__global__ void LaunchPhoton_Global(PhotonStruct* pd)
{
	unsigned long long PosData = blockIdx.x*NUM_THREADS_PER_BLOCK + threadIdx.x;
	//First element processed by the block
	if (PosData < num_photons_dc[0]){
	
		//	 LaunchPhoton(&pd[PosData], d_x[PosData], d_a[PosData]);
		pd[PosData].dx = 0.0f;
		pd[PosData].dy = 0.0f;
		pd[PosData].dz = 1.0f;
		pd[PosData].x	= 0.0f;
		pd[PosData].y	= 0.0f;
		pd[PosData].z	= 0.0;
		pd[PosData].layer	= 1;
		pd[PosData].Index = PosData;
		pd[PosData].weight	= (unsigned int)*start_weight_dc;

		//DeviceMem->p[begin + tx] = p;//incoherent!?
		
	}
	return;
}
__global__ void SetRandpram(curandState* devState){
	int id = threadIdx.x + blockIdx.x * blockDim.x;
	/* Each thread gets same seed, a different sequence number,
	no offset */
	curand_init(*dc_Seed, id, 0, &devState[id]);
}

__global__ void InitRng(MemStruct devMem,curandState* RndMakerglobal){
	curandState RndMakerLocal;
	RndMakerLocal = RndMakerglobal[threadIdx.x];
	unsigned long long* X = devMem.x;
	unsigned int* A = devMem.a;
	unsigned long long un64PosData = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int TmpRndH, TmpRndL;
	TmpRndH = curand(&RndMakerLocal);
	__syncthreads();
	TmpRndL = curand(&RndMakerLocal);
	unsigned long long TmpRAXH = ((unsigned long long)TmpRndH) << 32;
	X[un64PosData] = TmpRAXH | (unsigned long long)TmpRndL;
	__syncthreads();
	TmpRndL = curand(&RndMakerLocal);
	A[un64PosData] = TmpRndL;
	un64PosData += NUM_DIV_MAKE_RAND*NUM_THREADS_PER_BLOCK_MAKE_RAND;


		

	return;
}
__device__ double SpinTheta(unsigned long long int* x, unsigned int *a, double g){
	double cost;

	if (g == 0.0)
		cost = 2 * rand_MWC_co(x, a) - 1;
	else {
		double temp = (1 - g*g) / (1 - g + 2 * g*rand_MWC_co(x, a));
		cost = (1 + g*g - temp*temp) / (2 * g);
		if (cost < -1) cost = -1;
		else if (cost > 1) cost = 1;
	}
	return(cost);
}
__device__ void Hop(PhotonStruct* p,float s){
	p->x = s*p->dx;
	p->y = s*p->dy;
	p->z = s*p->dz;
}
__device__ void Spin(PhotonStruct* p, unsigned long long int* x, unsigned int *a, float g)
{
	float cost, sint;	// cosine and sine of the 
	// polar deflection angle theta. 
	float cosp, sinp;	// cosine and sine of the 
	// azimuthal angle psi. 
	float temp=2.1;

	float tempdir = p->dx;
	// Open CUDA Code
	//This is more efficient for g!=0 but of course less efficient for g==0
	//	temp = __fdividef((1.0f - (g)*(g)), (1.0f - (g)+2.0f*(g)*rand_MWC_co(x, a)));//Should be close close????!!!!!
	//	cost = __fdividef((1.0f + (g)*(g)-temp*temp), (2.0f*(g)));
	//	if (g == 0.0f)
	//		cost = 2.0f*rand_MWC_co(x, a) - 1.0f;//Should be close close??!!!!!

	// MIYAHIRA mcml SpinTheta()
	cost = SpinTheta(x,a,g);

	sint = sqrtf(1.0f - cost*cost);

	__sincosf(2.0f*PI*rand_MWC_co(x, a), &sinp, &cosp);// spin psi [0-2*PI)

	temp = sqrtf(1.0f - p->dz*p->dz);

	if (temp == 0.0f) //normal incident.
	{
		p->dx = sint*cosp;
		p->dy = sint*sinp;
		p->dz = copysignf(cost, p->dz*cost);	// copysign(a,b)==  a*SIGN(b) 
	}
	else // regular incident.
	{
		p->dx = __fdividef(sint*(p->dx*p->dz*cosp - p->dy*sinp), temp) + p->dx*cost;
		p->dy = __fdividef(sint*(p->dy*p->dz*cosp + tempdir*sinp), temp) + p->dy*cost;
		p->dz = -sint*cosp*temp + p->dz*cost;
	}

	//normalisation seems to be required as we are using floats! Otherwise the small numerical error will accumulate
	temp = rsqrtf(p->dx*p->dx + p->dy*p->dy + p->dz*p->dz);
	p->dx = p->dx*temp;
	p->dy = p->dy*temp;
	p->dz = p->dz*temp;

}// end Spin
__device__ unsigned int Reflect(PhotonStruct* p, int new_layer, unsigned long long* x, unsigned int* a)
{
	//Calculates whether the photon is reflected (returns 1) or not (returns 0)
	// Reflect() will also update the current photon layer (after transmission) and photon direction (both transmission and reflection)


	float n1 = layers_dc[p->layer].n;
	float n2 = layers_dc[new_layer].n;
	float r;
	float cos_angle_i = fabsf(p->dz);

	if (n1 == n2)//refraction index matching automatic transmission and no direction change
	{
		p->layer = new_layer;
		return 0u;
	}

	if (n1>n2 && n2*n2<n1*n1*(1 - cos_angle_i*cos_angle_i))//total internal reflection, no layer change but z-direction mirroring
	{
		p->dz *= -1.0f;
		return 1u;
	}

	if (cos_angle_i == 1.0f)//normal incident
	{
		r = __fdividef((n1 - n2), (n1 + n2));
		if (rand_MWC_co(x, a) <= r*r)
		{
			//reflection, no layer change but z-direction mirroring
			p->dz *= -1.0f;
			return 1u;
		}
		else
		{	//transmission, no direction change but layer change
			p->layer = new_layer;
			return 0u;
		}
	}
	else
	{
		//long and boring calculations of r
		float sinangle_i = sqrtf(1.0f - p->dz*p->dz);
		float sinangle_e = n1/n2*sinangle_i;
		float cosangle_e = sqrtf(1.0f - sinangle_e*sinangle_e);

		float cossumangle = (p->dz*cosangle_e) - sinangle_i*sinangle_e;
		float cosdiffangle = (p->dz*cosangle_e) + sinangle_i*sinangle_e;
		float sinsumangle = sinangle_i*cosangle_e + (p->z*sinangle_e);
		float sindiffangle = sinangle_i*cosangle_e - (p->z*sinangle_e);

		r = 0.5*sindiffangle*sindiffangle*__fdividef((cosdiffangle*cosdiffangle + cossumangle*cossumangle), (sinsumangle*sinsumangle*cosdiffangle*cosdiffangle));

	}
	//gives almost exactly the same results as the old MCML way of doing the calculation but does it slightly faster
	// save a few multiplications, calculate cos_angle_i^2;
	//float e = __fdividef(n1*n1, n2*n2)*(1.0f - cos_angle_i*cos_angle_i); //e is the sin square of the transmission angle
	//r = 2 * sqrtf((1.0f - cos_angle_i*cos_angle_i)*(1.0f - e)*e*cos_angle_i*cos_angle_i);//use r as a temporary variable
	//e = e + (cos_angle_i*cos_angle_i)*(1.0f - 2.0f*e);//Update the value of e
	//r = e*__fdividef((1.0f - e - r), ((1.0f - e + r)*(e + r)));//Calculate r	

	if (rand_MWC_co(x, a) <= r)
	{
		// Reflection, mirror z-direction!
		p->dz *= -1.0f;
		return 1u;
	}
	else
	{
		// Transmission, update layer and direction
		r = __fdividef(n1, n2);
		float e = r*r*(1.0f - cos_angle_i*cos_angle_i); //e is the sin square of the transmission angle
		p->dx *= r;
		p->dy *= r;
		p->dz = copysignf(sqrtf(1 - e), p->dz);
		p->layer = new_layer;
		return 0u;
	}
	p->rr = r;
}
__device__ unsigned int PhotonSurvive(PhotonStruct* p, unsigned long long* x, unsigned int* a)
{	//Calculate wether the photon survives (returns 1) or dies (returns 0)

	if (p->weight>WEIGHTI) return 1u; // No roulette needed
	if (p->weight == 0u) return 0u;	// Photon has exited slab, i.e. kill the photon

	if (rand_MWC_co(x, a)<CHANCE)
	{
		p->weight = __float2uint_rn(__fdividef((float)p->weight, CHANCE));
		return 1u;
	}

	//else
	return 0u;
}

//Device function to add an unsigned integer to an unsigned long long using CUDA Compute Capability 1.1
__device__ void AtomicAddULL(unsigned long long* address, unsigned long long add)
{
	if (atomicAdd((unsigned long long*)address, add) + add<add)
		atomicAdd(((unsigned long long*)address) + 1, 1u);
}
__device__ float rand_MWC_co(unsigned long long* x, unsigned int* a)
{
	float temp = 0.0;
	//Generate a random number [0,1)
	*x = (*x & 0xffffffffull)*(*a) + (*x >> 32);
	temp = __fdividef(__uint2float_rz((unsigned int)(*x)), (float)0x100000000);// The typecast will truncate the x so that it is 0<=x<(2^32-1),__uint2float_rz ensures a round towards zero since 32-bit floating point cannot represent all integers that large. Dividing by 2^32 will hence yield [0,1)
	return temp;
}//end __device__ rand_MWC_co
__device__ float rand_MWC_oc(unsigned long long* x, unsigned int* a)
{
	//Generate a random number (0,1]
	return 1.0f - rand_MWC_co(x, a);
}//end __device__ rand_MWC_oc


cCUDAMCML::cCUDAMCML(){

}
cCUDAMCML::~cCUDAMCML(){
}
void cCUDAMCML::RunOldCarnel(){
	dim3 dimGrid(NUM_GRID_PER_BLOCK);
	dim3 dimBlock(NUM_THREADS_PER_BLOCK);
	unsigned int threads_active_total = 1;
	int i = 0;
	while (*m_sHostMem.num_terminated_photons < m_simulations->number_of_photons)
	{
		i++;
		//run the kernel
		if (m_simulations->ignoreAdetection == 1){
			MCd<1> << <dimGrid, dimBlock >> >(m_sDeviceMem);
		}
		else{
			MCd<0> << <dimGrid, dimBlock >> >(m_sDeviceMem);
		}
		cudaThreadSynchronize(); // Wait for all threads to finish
		cudaError_t cudastat = cudaGetLastError(); // Check if there was an error

		// Copy thread_active from device to host
		cudaMemcpy(m_sHostMem.thread_active, m_sDeviceMem.thread_active, NUM_THREADS*sizeof(unsigned int), cudaMemcpyDeviceToHost);
		threads_active_total = 0;
		for (int ii = 0; ii < NUM_THREADS; ii++){
			threads_active_total += m_sHostMem.thread_active[ii];
		}

		cudaMemcpy(m_sHostMem.num_terminated_photons, m_sDeviceMem.num_terminated_photons, sizeof(unsigned int), cudaMemcpyDeviceToHost);
		cudaMemcpy(m_sHostMem.p, m_sDeviceMem.p, sizeof(PhotonStruct)*m_simulations->number_of_photons, cudaMemcpyDeviceToHost);
	}
	CopyDeviceToHostMem(&m_sHostMem, &m_sDeviceMem, m_simulations);
}
// 計算の中枢
int cCUDAMCML::MakeRandTableDev(){
	curandState *devStates;
	cudaError_t  cudastat;
	dim3 dimNumBlockRand(NUM_GRID_PER_BLOCK);
	dim3 dimNumThreadRand(NUM_THREADS_PER_BLOCK);
	cudaMalloc((void **)&devStates, NUM_THREADS * sizeof(curandState));
	// シード，初期値として用いる乱数配列の作成
	// MCMLの乱数生成に利用できない　⇒　ライブラリの都合上，スレッド数が制限されるため
	SetRandpram << < dimNumBlockRand, dimNumThreadRand >> > (devStates);
	cudastat = cudaGetLastError();	// Check if there was an error
	if (cudastat){
		return _ERR_GPU_SIM_RND_;
	}
	cudaThreadSynchronize();
	InitRng << < dimNumBlockRand, dimNumThreadRand >> > (m_sDeviceMem,devStates);
	// 検証用
	cudaMemcpy(m_sHostMem.a, m_sDeviceMem.a, NUM_THREADS * sizeof(unsigned int), cudaMemcpyDeviceToHost);
	cudaMemcpy(m_sHostMem.x, m_sDeviceMem.x, NUM_THREADS * sizeof(unsigned long long), cudaMemcpyDeviceToHost);
	cudastat = cudaGetLastError();	// Check if there was an error
	if (cudastat){
		return _ERR_GPU_SIM_RND_;
	}
	return 0;
}
int cCUDAMCML::InitPhoton(){


	dim3 dimNumBlock(19);
	dim3 dimNumThread(NUM_THREADS_PER_BLOCK);

	LaunchPhoton_Global << < dimNumBlock, dimNumThread >> > (m_sDeviceMem.p);
	cudaError_t cudastat = cudaGetLastError();	// Check if there was an error

	if (cudastat){
		return _ERR_GPU_SIM_LANCH_PHOTON_;
	}
	return 0;
}
int cCUDAMCML::DoOneSimulation(SimulationStruct* simulation)
{

	unsigned int threads_active_total = simulation->number_of_photons;
	unsigned int i;

	cudaError_t cudastat;

	int STAT = 0;
	// Start the clock

	dim3 dimNumBlock(NUM_GRID_PER_BLOCK);
	dim3 dimNumThread(NUM_THREADS_PER_BLOCK);
	int TotalP = 0;
	while (TotalP<simulation->number_of_photons){
		//run the kernel
		if (simulation->ignoreAdetection == 1){
			CalcMCGPU<1> << < dimNumBlock, dimNumThread >> >(m_sDeviceMem);
	
		}
		else{
			CalcMCGPU<0> << < dimNumBlock, dimNumThread >> >(m_sDeviceMem);
	
		}
		cudastat = cudaGetLastError(); // Check if there was an error
		if (cudastat){
			return _ERR_GPU_SIM_MCML_;
		}
		// 検証用
		cudastat = cudaMemcpy(m_sHostMem.p, m_sDeviceMem.p, NUM_THREADS * sizeof(PhotonStruct), cudaMemcpyDeviceToHost);
		cudastat = cudaMemcpy(m_sHostMem.thread_active, m_sDeviceMem.thread_active, NUM_THREADS * sizeof(unsigned int), cudaMemcpyDeviceToHost);
		cudastat = cudaMemcpy(m_sHostMem.num_terminated_photons, m_sDeviceMem.num_terminated_photons,sizeof(unsigned int), cudaMemcpyDeviceToHost);
		//	cudaThreadSynchronize();		// Wait for all threads to finish
		// 
		cudastat = cudaGetLastError(); // Check if there was an error
		if (cudastat){
			return _ERR_GPU_SIM_MEMCPY_;
		}
		for (int i = 0; i < NUM_THREADS; i++){
			if (m_sHostMem.thread_active[i] != 65535){
				TotalP += m_sHostMem.thread_active[i];
			}
		}
	}

	cudastat = cudaGetLastError(); // Check if there was an error
	if (cudastat){
		return _ERR_GPU_SIM_ANOTHER_;
	}


	CopyDeviceToHostMem(&m_sHostMem, &m_sDeviceMem, simulation);
	return _SUCCESS_GPU_SIM_;

}
int cCUDAMCML::InitMallocMem(SimulationStruct* sim){
	unsigned int State = 0;
	cudaError_t tmp;

	tmp = cudaMalloc((void**)&m_sDeviceMem, sizeof(MemStruct));
	if (tmp != cudaSuccess) {
		State |= 0x01;
	}
	tmp = cudaMalloc((void**)&m_sDeviceMem.p, (NUM_THREADS)*sizeof(PhotonStruct));
	if (tmp != cudaSuccess) {
		State |= 0x01;
	}
	tmp = cudaMalloc((void**)&m_sDeviceMem.x, (NUM_THREADS)*sizeof(unsigned long long));
	if (tmp != cudaSuccess) {
		State |= 0x02;
	}
	tmp = cudaMalloc((void**)&m_sDeviceMem.a, (NUM_THREADS)*sizeof(unsigned int));
	if (tmp != cudaSuccess) {
		State |= 0x04;
	}
	tmp = cudaMalloc((void**)&m_sDeviceMem.In_Ptr, sizeof(InputStruct));
	if (tmp != cudaSuccess) {
		State |= 0x08;
	}
	tmp = cudaMalloc((void**)&m_sDeviceMem.Out_Ptr, sizeof(OutStruct));
	if (tmp != cudaSuccess) {
		State |= 0x10;
	}
	tmp = cudaMalloc((void**)&m_sDeviceMem.thread_active, (NUM_THREADS)*sizeof(unsigned int));
	if (tmp != cudaSuccess) {
		State |= 0x20;
	}
	tmp = cudaMalloc((void**)&m_sDeviceMem.num_terminated_photons, sizeof(unsigned int));
	if (tmp != cudaSuccess) {
		State |= 0x40;
	}
	int rz_size = sim->det.nr*sim->det.nz;
	int ra_size = sim->det.nr*sim->det.na;
	tmp = cudaMalloc((void**)&m_sDeviceMem.A_rz, rz_size *sizeof(unsigned long long));
	if (tmp != cudaSuccess) {
		State |= 0x80;
	}
	tmp = cudaMalloc((void**)&m_sDeviceMem.Rd_ra, ra_size*sizeof(unsigned long long));
	if (tmp != cudaSuccess) {
		State |= 0x100;
	}
	tmp = cudaMalloc((void**)&m_sDeviceMem.Tt_ra, ra_size*sizeof(unsigned long long));
	if (tmp != cudaSuccess) {
		State |= 0x200;
	}
	tmp = cudaMalloc((void**)&m_Out.Rd_ra, ra_size*sizeof(double));
	if (tmp != cudaSuccess) {
		State |= 0x400;
	}
	tmp = cudaMalloc((void**)&m_Out.Rd_p, ra_size*sizeof(double));
	if (tmp != cudaSuccess) {
		State |= 0x800;
	}

	// Allocate p on the device!!
	// Allocate A_rz on host and device
	m_sHostMem.p = new PhotonStruct			[NUM_THREADS];
	m_sHostMem.x = new unsigned long long	[NUM_THREADS];
	m_sHostMem.a = new unsigned int			[NUM_THREADS];
	if ((m_sHostMem.x != NULL) && (m_sHostMem.a != NULL)){
		State |= 0x00200000;
	}
	m_sHostMem.A_rz = new unsigned long long [rz_size];
	if (m_sHostMem.A_rz == NULL){
		State |= 0x00010000;
	}
	m_sHostMem.Rd_ra = new unsigned long long[ra_size];
	if (m_sHostMem.Rd_ra == NULL){
		State |= 0x00020000;
	}
	m_sHostMem.Tt_ra = new unsigned long long[ra_size];
	if (m_sHostMem.Tt_ra == NULL){
		State |= 0x00040000;
	}
	// Allocate thread_active on the device and host
	m_sHostMem.thread_active = new unsigned int[sim->number_of_photons];
	if (m_sHostMem.thread_active == NULL){
		State |= 0x00080000;
	}

	m_sHostMem.num_terminated_photons = new unsigned int[1];
	if (m_sHostMem.num_terminated_photons == NULL){
		State |= 0x00100000;
	}
	*m_sHostMem.num_terminated_photons = 0;

	m_sHostMem.Out_Ptr->Rd_ra = new double[ra_size];
	if (m_sHostMem.Out_Ptr->Rd_ra == NULL){
		State |= 0x00200000;
	}
	m_sHostMem.Out_Ptr->Rd_p = new double[rz_size];
	if (m_sHostMem.Out_Ptr->Rd_p == NULL){
		State |= 0x00400000;
	}
	m_Out.OPL = new double,sizeof(double);
	if (m_Out.OPL == NULL){
		State |= 0x00800000;
	}
	m_sHostMem.Out_Ptr->L = new double ,sizeof(double);
	if (m_Out.L == NULL){
		State |= 0x001000000;
	}
	m_sHostMem.Out_Ptr->opl = new double ,sizeof (double);
	if (m_sHostMem.Out_Ptr->Rd_p == NULL){
		State |= 0x02000000;
	}


	return State;
}
void cCUDAMCML::CopyDeviceToHostMem(MemStruct* HostMem, MemStruct* DeviceMem, SimulationStruct* sim)
{ //Copy data from Device to Host memory

	int rz_size = sim->det.nr*sim->det.nz;
	int ra_size = sim->det.nr*sim->det.na;

	// Copy A_rz, Rd_ra and Tt_ra
	cudaMemcpy(HostMem->A_rz, DeviceMem->A_rz, rz_size*sizeof(unsigned long long), cudaMemcpyDeviceToHost);
	cudaMemcpy(HostMem->Rd_ra, DeviceMem->Rd_ra, ra_size*sizeof(unsigned long long), cudaMemcpyDeviceToHost);
	cudaMemcpy(HostMem->Tt_ra, DeviceMem->Tt_ra, ra_size*sizeof(unsigned long long), cudaMemcpyDeviceToHost);

	//Also copy the state of the RNG's
	cudaMemcpy(HostMem->p, DeviceMem->p, NUM_THREADS *sizeof(PhotonStruct), cudaMemcpyDeviceToHost);

	return ;
}
int cCUDAMCML::CopyHostToDeviceMem(MemStruct* HostMem, MemStruct* DeviceMem, SimulationStruct* sim){
	// Allocate x and a on the device (For MWC RNG)
	cudaError_t tmp;
	int State = 0;
	tmp = cudaMemcpy(DeviceMem->x, HostMem->x, NUM_THREADS*sizeof(unsigned long long), cudaMemcpyHostToDevice);
	if (tmp != cudaSuccess) {
		State |= 0x40;
	}


	tmp = cudaMemcpy(DeviceMem->a, HostMem->a, NUM_THREADS*sizeof(unsigned int), cudaMemcpyHostToDevice);
	if (tmp != cudaSuccess) {
		State |= 0x80;
	}
	tmp = cudaMemcpy(DeviceMem->thread_active, HostMem->thread_active, NUM_THREADS*sizeof(unsigned int), cudaMemcpyHostToDevice);
	if (tmp != cudaSuccess) {
		State |= 0x200;
	}
	tmp = cudaMemcpy(DeviceMem->num_terminated_photons, HostMem->num_terminated_photons, sizeof(unsigned int), cudaMemcpyHostToDevice);
	if (tmp != cudaSuccess) {
		State |= 0x800;
	}
	return State;
}
int cCUDAMCML::InitDCMem(SimulationStruct* sim)
{
	cudaError_t tmp;
	int State = 0;
	// Copy det-data to constant device memory
	tmp = cudaMemcpyToSymbol(det_dc, &(sim->det), sizeof(DetStruct));
	if (tmp != cudaSuccess) {
		State |= 0x1;
	}

	// Copy num_photons_dc to constant device memory
	tmp = cudaMemcpyToSymbol(n_layers_dc, &(sim->n_layers), sizeof(unsigned int));
	if (tmp != cudaSuccess) {
		State |= 0x2;
	}

	// Copy start_weight_dc to constant device memory
	tmp = cudaMemcpyToSymbol(start_weight_dc, &(sim->start_weight), sizeof(unsigned long long));
	if (tmp != cudaSuccess) {
		State |= 0x4;
	}

	// Copy layer data to constant device memory
	tmp = cudaMemcpyToSymbol(layers_dc, sim->layers, (sim->n_layers + 2)*sizeof(LayerStruct));
	if (tmp != cudaSuccess) {
		State |= 0x8;
	}

	// Copy num_photons_dc to constant device memory
	tmp = cudaMemcpyToSymbol(num_photons_dc, &(sim->number_of_photons), sizeof(unsigned long long));
	if (tmp != cudaSuccess) {
		State |= 0x10;
	}
	// Copy num_photons_dc to constant device memory
	tmp = cudaMemcpyToSymbol(dc_Seed, &(sim->Seed), sizeof(unsigned int));
	if (tmp != cudaSuccess) {
		State |= 0x20;
	}
	


	return State;

}
int cCUDAMCML::InitMemStructs(MemStruct* HostMem, MemStruct* DeviceMem, SimulationStruct* sim)
{
	int rz_size, ra_size;
	cudaError_t tmp;
	rz_size = sim->det.nr*sim->det.nz;
	ra_size = sim->det.nr*sim->det.na;
	int Total = 0;

	// Allocate p on the device!!
	cudaMalloc((void**)&DeviceMem->p, NUM_THREADS*sizeof(PhotonStruct));
	Total += NUM_THREADS*sizeof(PhotonStruct);
	// Allocate A_rz on host and device
	HostMem->A_rz = new(unsigned long long)(rz_size*sizeof(unsigned long long));
	if (HostMem->A_rz == NULL){ 
		printf("Error allocating HostMem->A_rz"); 
		exit(1); 
	}
	cudaMalloc((void**)&DeviceMem->A_rz, rz_size*sizeof(unsigned long long));
	Total += rz_size*sizeof(unsigned long long);
	cudaMemset(DeviceMem->A_rz, 0, rz_size*sizeof(unsigned long long));

	// Allocate Rd_ra on host and device
	HostMem->Rd_ra = new(unsigned long long)(ra_size*sizeof(unsigned long long));
	if (HostMem->Rd_ra == NULL){ 
		printf("Error allocating HostMem->Rd_ra"); 
		exit(1);
	}
	cudaMalloc((void**)&DeviceMem->Rd_ra, ra_size*sizeof(unsigned long long));
	Total += ra_size*sizeof(unsigned long long);
	cudaMemset(DeviceMem->Rd_ra, 0, ra_size*sizeof(unsigned long long));

	// Allocate Tt_ra on host and device
	HostMem->Tt_ra = new(unsigned long long)(ra_size*sizeof(unsigned long long));
	if (HostMem->Tt_ra == NULL){ printf("Error allocating HostMem->Tt_ra"); exit(1); }
	cudaMalloc((void**)&DeviceMem->Tt_ra, ra_size*sizeof(unsigned long long));
	Total += ra_size*sizeof(unsigned long long);
	cudaMemset(DeviceMem->Tt_ra, 0, ra_size*sizeof(unsigned long long));


	// Allocate x and a on the device (For MWC RNG)
	cudaMalloc((void**)&DeviceMem->x, NUM_THREADS*sizeof(unsigned long long));
	Total += NUM_THREADS*sizeof(unsigned long long);

	cudaMemcpy(DeviceMem->x, HostMem->x, NUM_THREADS*sizeof(unsigned long long), cudaMemcpyHostToDevice);
	tmp = cudaMalloc((void**)&DeviceMem->a, NUM_THREADS*sizeof(unsigned int));
	Total += NUM_THREADS*sizeof(unsigned long long);

	tmp = cudaMemcpy(DeviceMem->a, HostMem->a, NUM_THREADS*sizeof(unsigned int), cudaMemcpyHostToDevice);


	// Allocate thread_active on the device and host
	HostMem->thread_active = new unsigned int (NUM_THREADS*sizeof(unsigned int));
	if (HostMem->thread_active == NULL){ printf("Error allocating HostMem->thread_active"); exit(1); }
	for (int i = 0; i < NUM_THREADS; i++){
		HostMem->thread_active[i] = 1u;
	}

	Total += NUM_THREADS*sizeof(unsigned int);
	tmp = cudaMalloc((void**)&(DeviceMem->thread_active), NUM_THREADS*sizeof(unsigned int));

	tmp = cudaMemcpy(DeviceMem->thread_active, HostMem->thread_active, NUM_THREADS*sizeof(unsigned int), cudaMemcpyHostToDevice);


	//Allocate num_launched_photons on the device and host
	HostMem->num_terminated_photons = new(unsigned int)(sizeof(unsigned int));
	if (HostMem->num_terminated_photons == NULL){ printf("Error allocating HostMem->num_terminated_photons"); exit(1); }
	*HostMem->num_terminated_photons = 0;

	cudaMalloc((void**)&DeviceMem->num_terminated_photons, sizeof(unsigned int));
	cudaMemcpy(DeviceMem->num_terminated_photons, HostMem->num_terminated_photons, sizeof(unsigned int), cudaMemcpyHostToDevice);

	return 1;
}
int cCUDAMCML::InitContentsMem(SimulationStruct* sim)
{
	MemStruct* DeviceMem = &m_sDeviceMem;
	MemStruct* HostMem = &m_sHostMem;

	int State=0;
	int rz_size, ra_size;
	cudaError_t tmp;
	rz_size = sim->det.nr*sim->det.nz;
	ra_size = sim->det.nr*sim->det.na;

	tmp = cudaMemset(DeviceMem->A_rz, 0, rz_size*sizeof(unsigned long long));
	if (tmp != cudaSuccess) {
		State |= 0x02;
	}

	tmp = cudaMemset(DeviceMem->Rd_ra, 0, ra_size*sizeof(unsigned long long));
	if (tmp != cudaSuccess) {
		State |= 0x08;
	}

	tmp = cudaMemset(DeviceMem->Tt_ra, 0, ra_size*sizeof(unsigned long long));
	if (tmp != cudaSuccess) {
		State |= 0x20;
	}

	PhotonStruct TmpPS;

	HostMem->p->x = 0;
	HostMem->p->y = 0;
	HostMem->p->z = 0;
	HostMem->p->dx = 0;
	HostMem->p->dy = 0;
	HostMem->p->dz = 0;
	HostMem->p->weight = 0;
	HostMem->p->layer = 0;

	tmp = cudaMemset(DeviceMem->p, 0, NUM_THREADS *sizeof(PhotonStruct));
	if (tmp != cudaSuccess) {
		State |= 0x20;
	}

	tmp = cudaMemcpy(DeviceMem->x, HostMem->x, NUM_THREADS *sizeof(unsigned long long), cudaMemcpyHostToDevice);
	if (tmp != cudaSuccess) {
		State |= 0x40;
	}


	tmp = cudaMemcpy(DeviceMem->a, HostMem->a, NUM_THREADS*sizeof(unsigned int), cudaMemcpyHostToDevice);
	if (tmp != cudaSuccess) {
		State |= 0x80;
	}

	for (int i = 0; i < sim->number_of_photons; i++){
		HostMem->thread_active[i] = 1u;
	}



	tmp = cudaMemcpy(DeviceMem->thread_active, HostMem->thread_active, NUM_THREADS*sizeof(unsigned int), cudaMemcpyHostToDevice);
	if (tmp != cudaSuccess) {
		State |= 0x200;
	}



	tmp = cudaMemcpy(DeviceMem->num_terminated_photons, HostMem->num_terminated_photons, sizeof(unsigned int), cudaMemcpyHostToDevice);
	if (tmp != cudaSuccess) {
		State |= 0x800;
	}

	return State;
}

void cCUDAMCML::FreeMemStructs(MemStruct* HostMem, MemStruct* DeviceMem)
{

	cudaFree(DeviceMem->p);
	cudaFree(DeviceMem->x);
	cudaFree(DeviceMem->a);
	cudaFree(DeviceMem->thread_active);
	cudaFree(DeviceMem->num_terminated_photons);
	cudaFree(DeviceMem->A_rz);
	cudaFree(DeviceMem->Rd_ra);
	cudaFree(DeviceMem->Tt_ra);
	cudaFree(DeviceMem);
	delete[] HostMem->p;
	delete[] HostMem->x;
	delete[] HostMem->a;
	delete[] HostMem->thread_active;
	delete[] HostMem->num_terminated_photons;
	delete[] HostMem->Reserve;
	delete[] HostMem->A_rz;
	delete[] HostMem->Rd_ra;
	delete[] HostMem->Tt_ra;


}

void cCUDAMCML::FreeSimulationStruct(SimulationStruct* sim, int nRun)
{
	FreeMemStructs(&m_sHostMem, &m_sDeviceMem);
	for (int i = 0; i < nRun; i++){
		delete sim[i].layers;
	}
	delete[] sim;
	//cudaDeviceReset();
}

void cCUDAMCML::FreeFailedSimStrct(SimulationStruct* Sim, int nRun)
{
	for (int i = 0; i < nRun; i++){
		delete Sim[i].layers;
	}
	delete[] Sim;

}


bool cCUDAMCML::CheckGPU(){
	int deviceCount;
	cudaGetDeviceCount(&deviceCount);
	if (deviceCount == 0){
		return 0;
	}
	cudaSetDevice(0);
	cudaGetDeviceProperties(&m_sDevProp, 0);
	return 1;
}
void cCUDAMCML::InitGPUStat(){
	m_ProcessTime = 0;
	m_un64Membyte = 0;
	m_un64NumPhoton = 0;
	m_un64PrcsDataNum = 0;
	cudaDeviceReset();
}
/***********************************************************
*	Record the photon weight exiting the first layer(uz<0),
*	no matter whether the layer is glass or not, to the
*	reflection array.
*
*	Update the photon weight as well.
****/
extern "C"{
	__host__ __device__ void RemodelRecordR(MemStruct  DeviceMem, PhotonStruct *p)
	{
		InputStruct *In_Ptr = DeviceMem.In_Ptr;
		OutStruct	*Out_Ptr = DeviceMem.Out_Ptr;
		double	Refl = p->rr;
		double x = p->x;
		double y = p->y;
		double dx = p->dx;
		double dy = p->dy;
		double dz = p->dz;
		double t;
		double r;
		double r1 = In_Ptr->r;
		double t1, t2, t3;
		short  it, ia;	/* index to r & angle. */
		double itd, iad;	/* LW 5/20/98. To avoid out of short range.*/
		short  nl = In_Ptr->num_layers;
		short	 l;
		int	 n = Out_Ptr->p1;
		int id;

		r = sqrt(x*x + y*y);

		if (r >= r1 && r <= (r1 + 0.1*r1))
		{
			if (y >= 0)
				t1 = atan2(y, x) * 180 / PI;
			else
				t1 = 360 + atan2(y, x) * 180 / PI;

			if (dy <= 0)
				t2 = atan2(-dy, -dx) * 180 / PI;
			else
				t2 = 360 + atan2(-dy, -dx) * 180 / PI;

			t3 = t2 - t1;
			if (t3 < 0)
				t = 360 + t3;
			else
				t = t3;

			itd = (short)(t / In_Ptr->dt);
			if (itd > In_Ptr->nr - 1) it = In_Ptr->nr - 1;
			else it = itd;

			iad = (short)(acos(-dz) * 180 / PI / In_Ptr->da);
			if (iad > In_Ptr->na - 1) ia = In_Ptr->na - 1;
			else ia = iad;

			Out_Ptr->Rd_ra[In_Ptr->nr*ia + it] += p->weight*(1.0 - Refl);		/* 各天頂角・各方位角の光子ウェイトの記録 */
			Out_Ptr->Rd_p[In_Ptr->nr*ia + it] += 1;							/* 各天頂角・各方位角の光子数の記録 */
			Out_Ptr->P += p->weight*(1.0 - Refl);

			for (l = 1; l <= nl; l++)
				Out_Ptr->L[l] += Out_Ptr->OPL[l] * p->weight*(1.0 - Refl);		/* 受光エリアに入った光子の光路長の記録 */

			Out_Ptr->p1 += 1;
		}

		p->weight *= Refl;
	}
	//__host__ __device__ void RecordR(double			Refl,	/* reflectance. */
	//	InputStruct  *	In_Ptr,
	//	PhotonStruct *	p,
	//	OutStruct *	Out_Ptr)
	//{
	//	double x = p->x;
	//	double y = p->y;
	//	double dx = p->dx;
	//	double dy = p->dy;
	//	double dz = p->dz;
	//	double t;
	//	double r;
	//	double r1 = In_Ptr->r;
	//	double t1, t2, t3;
	//	short  it, ia;	/* index to r & angle. */
	//	double itd, iad;	/* LW 5/20/98. To avoid out of short range.*/
	//	short  nl = In_Ptr->num_layers;
	//	short	 l;
	//	int	 n = Out_Ptr->p1;
	//
	//	r = sqrt(x*x + y*y);
	//
	//	if (r >= r1 && r <= (r1 + 0.1*r1))
	//	{
	//		if (y >= 0)
	//			t1 = atan2(y, x) * 180 / PI;
	//		else
	//			t1 = 360 + atan2(y, x) * 180 / PI;
	//
	//		if (dy <= 0)
	//			t2 = atan2(-dy, -dx) * 180 / PI;
	//		else
	//			t2 = 360 + atan2(-dy, -dx) * 180 / PI;
	//
	//		t3 = t2 - t1;
	//		if (t3<0)
	//			t = 360 + t3;
	//		else
	//			t = t3;
	//
	//		itd = (short)(t / In_Ptr->dt);
	//		if (itd>In_Ptr->nr - 1) it = In_Ptr->nr - 1;
	//		else it = itd;
	//
	//		iad = (short)(acos(-dz) * 180 / PI / In_Ptr->da);
	//		if (iad>In_Ptr->na - 1) ia = In_Ptr->na - 1;
	//		else ia = iad;
	//
	//		Out_Ptr->Rd_ra[it][ia] += p->weight*(1.0 - Refl);		/* 各天頂角・各方位角の光子ウェイトの記録 */
	//		Out_Ptr->Rd_p[it][ia] += 1;							/* 各天頂角・各方位角の光子数の記録 */
	//		Out_Ptr->P += p->weight*(1.0 - Refl);
	//
	//		for (l = 1; l <= nl; l++)
	//			Out_Ptr->L[l] += Out_Ptr->OPL[l] * p->weight*(1.0 - Refl);		/* 受光エリアに入った光子の光路長の記録 */
	//
	//		Out_Ptr->p1 += 1;
	//	}
	//
	//	p->weight *= Refl;
	//}
	__host__ __device__ void InitOutputData(MemStruct deviceMem, InputStruct In_Parm,
		OutStruct * Out_Ptr)
	{
		short nr = In_Parm.nr;
		short na = In_Parm.na;
		short nl = In_Parm.num_layers;
		/* remember to use nl+2 because of 2 for ambient. */

		if (nr <= 0 || na <= 0 || nl <= 0)
			//nrerror("Wrong grid parameters.\n");

			/* Init pure numbers. */
			Out_Ptr->Rsp = 0.0;

		/* Allocate the arrays and the matrices. */
		Out_Ptr->Rd_ra = AllocMatrix(0, nr - 1, 0, na - 1);
		Out_Ptr->Rd_p = AllocMatrix(0, nr - 1, 0, na - 1);

		Out_Ptr->OPL = AllocVector(0, nl + 1);
		Out_Ptr->L = AllocVector(0, nl + 1);
		Out_Ptr->opl = AllocVector(0, nl + 1);
	}
	void ReportResult(InputStruct In_Parm, OutStruct Out_Parm)
	{
		char time_report[STR_LEN];

		strcpy(time_report, " Simulation time of this run.");
		PunchTime(1, time_report);

		SumScaleResult(In_Parm, &Out_Parm);
		WriteResult(In_Parm, Out_Parm, time_report);
	}
	__host__ __device__ time_t PunchTime(char F, char *Msg)
	{
#if GNUCC
		return(0);
#else
		static clock_t ut0;	/* user time reference. */
		static time_t  rt0;	/* real time reference. */
		double secs;
		char s[STR_LEN];

		if (F == 0) {
			ut0 = clock();
			rt0 = time(NULL);
			return(0);
		}
		else if (F == 1)  {
			secs = (clock() - ut0) / (double)CLOCKS_PER_SEC;
			if (secs < 0) secs = 0;	/* clock() can overflow. */
			sprintf(s, "User time: %8.0lf sec = %8.2lf hr.  %s\n",
				secs, secs / 3600.0, Msg);
			puts(s);
			strcpy(Msg, s);
			return(difftime(time(NULL), rt0));
		}
		else if (F == 2) return(difftime(time(NULL), rt0));
		else return(0);
#endif
	}
	__host__ __device__ void SumScaleResult(InputStruct In_Parm, OutStruct * Out_Ptr)
	{
		CalOPL_SD(In_Parm, Out_Ptr);
	}
	__host__ void WriteResult(InputStruct In_Parm,
		OutStruct Out_Parm,
		char * TimeReport)
	{
		FILE *file;

		file = fopen(In_Parm.out_fname, "w");
		//	if (file == NULL) nrerror("Cannot open file to write.\n");

		if (toupper(In_Parm.out_fformat) == 'A')
			WriteVersion(file, "A1");
		else
			WriteVersion(file, "B1");

		fprintf(file, "# %s", TimeReport);
		fprintf(file, "\n");

		WriteInParm(file, In_Parm);
		/* reflectance, absorption, transmittance. */

		/* 1D arrays. */

		/* 2D arrays. */
		WriteRd_ra(file, In_Parm.nr, In_Parm.na, Out_Parm);
		WriteRd_p(file, In_Parm.nr, In_Parm.na, Out_Parm);
		WriteOPL(file, In_Parm.num_layers, Out_Parm);

		fclose(file);
	}
	__device__ __host__ void CalOPL_SD(InputStruct In_Parm, OutStruct * Out_Ptr)
	{
		short l;
		short	nl = In_Parm.num_layers;

		for (l = 1; l <= nl; l++)
			Out_Ptr->opl[l] = Out_Ptr->L[l] / Out_Ptr->P;		/* 第n層に入った光子の光路長の平均 */
	}

	__device__ __host__ double *AllocVector(short nl, short nh)
	{
		double *v;
		short i;

		v = (double *)malloc((unsigned)(nh - nl + 1)*sizeof(double));
		//	if (!v) nrerror("allocation failure in vector()");

		v -= nl;
		for (i = nl; i <= nh; i++) v[i] = 0.0;	/* init. */
		return v;
	}
	__host__ void WriteVersion(FILE *file, char *Version)
	{
		fprintf(file,
			"%s \t# Version number of the file format.\n\n",
			Version);
		fprintf(file, "####\n# Data categories include: \n");
		fprintf(file, "# InParm, RAT, \n");
		fprintf(file, "# Rd_ta, \n####\n\n");
	}
	/***********************************************************
	*	Write the input parameters to the file.
	****/
	void WriteInParm(FILE *file, InputStruct In_Parm)
	{
		short i;

		fprintf(file,
			"InParm \t\t\t# Input parameters. cm is used.\n");

		fprintf(file,
			"%s \tA\t\t# output file name, ASCII.\n",
			In_Parm.out_fname);
		fprintf(file,
			"%ld \t\t\t# No. of photons\n", In_Parm.num_photons);
		fprintf(file,
			"%.2lf \t\t\t# No. of SD distance\n", In_Parm.r);

		fprintf(file,
			"%G\t\t\t\t# dt [cm]\n", In_Parm.dt);
		fprintf(file, "%hd\t%hd\t\t# No. of dt, da.\n\n",
			In_Parm.nr, In_Parm.na);

		fprintf(file,
			"%hd\t\t\t\t\t# Number of layers\n",
			In_Parm.num_layers);
		fprintf(file,
			"#n\tmua\tmus\tg\td\t# One line for each layer\n");
		fprintf(file,
			"%G\t\t\t\t\t# n for medium above\n",
			In_Parm.layerspecs[0].n);
		for (i = 1; i <= In_Parm.num_layers; i++)  {
			LayerStruct s;
			s = In_Parm.layerspecs[i];
			fprintf(file, "%G\t%G\t%G\t%G\t%G\t# layer %hd\n",
				s.n, s.mua, s.mutr, s.g, s.z_max - s.z_min, i);
		}
		fprintf(file, "%G\t\t\t\t\t# n for medium below\n\n",
			In_Parm.layerspecs[i].n);
	}
	__host__ void WriteRd_ra(FILE * file,
		short Nr,
		short Na,
		OutStruct Out_Parm)
	{
		short it, ia;

		fprintf(file,
			"%s\n%s\n%s\n%s\n%s\n%s\n",	/* flag. */
			"# Rd[theta][angle]. [1/(cm2sr)].",
			"# Rd[0][0], [0][1],..[0][na-1]",
			"# Rd[1][0], [1][1],..[1][na-1]",
			"# ...",
			"# Rd[nt-1][0], [nt-1][1],..[nt-1][na-1]",
			"Rd_ta");

		for (it = 0; it < Nr; it++)
		{
			for (ia = 0; ia < Na; ia++)
			{
				fprintf(file, "%12.4E,", Out_Parm.Rd_ra[it + Nr*ia]);
				if ((it*Na + ia + 1) % 9 == 0) fprintf(file, "\n");
			}
		}
		fprintf(file, "\n");
	}

	/***********************************************************
	*	1 number each line.
	****/

	__host__ void WriteRd_p(FILE * file,
		short Nr,
		short Na,
		OutStruct Out_Parm)
	{
		short it, ia;

		fprintf(file,
			"%s\n%s\n%s\n%s\n%s\n%s\n",	/* flag. */
			"# Rd[theta][angle]. [1/(cm2sr)].",
			"# Rd[0][0], [0][1],..[0][na-1]",
			"# Rd[1][0], [1][1],..[1][na-1]",
			"# ...",
			"# Rd[nt-1][0], [nt-1][1],..[nt-1][na-1]",
			"Rd_p");

		for (it = 0; it < Nr; it++)
		{
			for (ia = 0; ia < Na; ia++)
			{
				fprintf(file, "%12.4E,", Out_Parm.Rd_p[it + ia*Nr]);
				if ((it*Na + ia + 1) % 9 == 0) fprintf(file, "\n");
			}
		}
		fprintf(file, "photon number");
		fprintf(file, "%ld\n", Out_Parm.p1);
		fprintf(file, "\n");
	}

	/***********************************************************
	*	1 number each line.
	****/
	__host__ void WriteOPL(FILE * file,
		short nl,
		OutStruct Out_Parm)
	{
		short l;

		for (l = 1; l <= nl; l++)
		{
			fprintf(file, "The %d layer\n", l);
			fprintf(file, "%12.4E\n", Out_Parm.opl[l]);	/* 平均光路長の書き込み */
			fprintf(file, "\n");
		}
	}
	__host__ __device__ double *AllocMatrix(short nrl, short nrh, short ncl, int nch)
	{
		long i, j;
		double *m;

		m = (double *)malloc((unsigned)(nrh - nrl + 1)*(nch - ncl + 1) *sizeof(double*));
		//if (!m) nrerror("allocation failure 1 in matrix()");
		//m -= nrl;

		//for (i = nrl; i <= nrh; i++) {
		//m[i] = (double *)malloc((unsigned)(nch - ncl + 1) *sizeof(double));
		//if (!m[i]) nrerror("allocation failure 2 in matrix()");
		//m[i] -= ncl;
		//}

		for (i = nrl; i <= nrh; i++)
			for (j = ncl; j <= nch; j++) m[i*(nrh - nrl + 1) + j] = 0.0;
		return m;
	}
}