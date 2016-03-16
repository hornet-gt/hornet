
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>


#include <unordered_map>
#include <algorithm>

#include "main.h"
#include "update.hpp"


using namespace std;


// __global__ void deviceUpdatesSweep1(cuStinger* custing, BatchUpdate* bu,int32_t updatesPerBlock);
// __global__ void deviceUpdatesSweep2(cuStinger* custing, BatchUpdate* bu,int32_t updatesPerBlock);
// __global__ void deviceRemoveInsertedDuplicates(cuStinger* custing, BatchUpdate* bu,int32_t dupsPerBlock);


int32_t elementsPerVertexOverLimit(int32_t elements, int32_t overLimit){
	int32_t eleCount = elements+overLimit;
	if(eleCount==0)
		eleCount=1;
	else if(eleCount < 5)
		eleCount*=2;
	else
		eleCount*=1.5;
	return eleCount;
}



void reAllocateMemoryAfterSweep1(cuStinger &custing, BatchUpdate &bu)
{
	int32_t sum=0, *tempsrc=bu.getHostSrc(),*tempdst=bu.getHostDst();
	int32_t *incomplete = bu.getHostIndIncomplete();	
	int32_t incCount = bu.getHostIncCount();

	unordered_map <int32_t, int32_t> h_hmap;

	int32_t* requireUpdates=(int32_t*)allocHostArray(bu.getHostBatchSize(), sizeof(int32_t));
	int32_t* overLimit=(int32_t*)allocHostArray(bu.getHostBatchSize(), sizeof(int32_t));

	for (int32_t i=0; i<incCount; i++){
		int32_t temp = tempsrc[incomplete[i]];
		h_hmap[temp]++;

		// if (temp==536954)
		// 	cout << "CPU 536954 " << tempdst[incomplete[i]] << endl;

	}

	int countUnique=0;
	for (int32_t i=0; i<incCount; i++){
		int32_t temp = tempsrc[incomplete[i]];
		if(h_hmap[temp]!=0){
			requireUpdates[countUnique]=temp;
			overLimit[countUnique]=h_hmap[temp];
			countUnique++;
			h_hmap[temp]=0;
		}
	}

	// sort(requireUpdates, requireUpdates + countUnique);

	for (int32_t i=0; i<countUnique; i++){
		int32_t tempVertex = requireUpdates[i];
		int32_t newMax = elementsPerVertexOverLimit(custing.h_max[tempVertex] ,overLimit[i]);
		int32_t* tempAdjacency = (int32_t*)allocDeviceArray(newMax, sizeof(int32_t));
		

		// if (tempVertex==536954)
		// cout << "Hello " << tempVertex << ", " << custing.h_max[tempVertex] << ", " << overLimit[i] << ", " << newMax << endl;
		copyArrayDeviceToHost(custing.h_adj[tempVertex], tempAdjacency, custing.h_max[tempVertex], sizeof(int32_t));

		freeDeviceArray(custing.h_adj[tempVertex]);
		custing.h_max[tempVertex] = newMax;
		custing.h_adj[tempVertex] = tempAdjacency;
	}
	custing.copyHostToDevice();

	freeHostArray(requireUpdates);
	freeHostArray(overLimit);

}