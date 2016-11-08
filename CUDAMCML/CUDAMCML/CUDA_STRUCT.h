
// ç≈ëÂílóp
__device__ struct MaxState{
	int MaxRzSize;
	int MaxRaSize;
};


// TYPEDEFS
__device__ struct LayerStruct {
	float z_min;		// Layer z_min [cm]
	float z_max;		// Layer z_max [cm]
	float mutr;			// Reciprocal mu_total [cm]
	float mua;			// Absorption coefficient [1/cm]
	float g;			// Anisotropy factor [-]
	float n;			// Refractive index [-]
};

__device__ struct PhotonStruct{
	float x;		// Global x coordinate [cm]
	float y;		// Global y coordinate [cm]
	float z;		// Global z coordinate [cm]
	float dx;		// (Global, normalized) x-direction
	float dy;		// (Global, normalized) y-direction
	float dz;		// (Global, normalized) z-direction
	unsigned long long weight;			// Photon weight
	int layer;				// Current layer

	__device__ PhotonStruct& operator = (const PhotonStruct& X){
		x = X.x;
		y = X.y;
		z = X.z;
		dx = X.dx;
		dy = X.dy;
		dz = X.dz;
		weight = X.weight;
		layer = X.layer;
		return *this;
	}
};

__device__ struct DetStruct{
	float dr;		// Detection grid resolution, r-direction [cm]
	float dz;		// Detection grid resolution, z-direction [cm]

	int na;			// Number of grid elements in angular-direction [-]
	int nr;			// Number of grid elements in r-direction
	int nz;			// Number of grid elements in z-direction

};


__device__ struct SimulationStruct{
	unsigned long number_of_photons;
	int ignoreAdetection;
	unsigned int n_layers;
	unsigned long long start_weight;
	char outp_filename[STR_LEN];
	char inp_filename[STR_LEN];
	char AorB;
	unsigned int  nDivedSimNum;
	long begin, end;
	DetStruct det;
	LayerStruct* layers;
};


__device__ struct MemStruct{
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
