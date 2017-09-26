/**
 * @internal
 * @author Oded Green                                                  <br>
 *         Georgia Institute of Technology, Computational Science and Engineering                   <br>
 *         ogreen@gatech.edu
 * @date August, 2017
 * @version v2
 *
 * @copyright Copyright © 2017 cuStinger. All rights reserved.
 *
 * @license{<blockquote>
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * * Redistributions of source code must retain the above copyright notice, this
 *   list of conditions and the following disclaimer.
 * * Redistributions in binary form must reproduce the above copyright notice,
 *   this list of conditions and the following disclaimer in the documentation
 *   and/or other materials provided with the distribution.
 * * Neither the name of the copyright holder nor the names of its
 *   contributors may be used to endorse or promote products derived from
 *   this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 * </blockquote>}
 *
 * @file
 */

/*
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <inttypes.h>

#include <math.h>
*/

#include "Dynamic/KatzCentrality/katz.cuh"

//#include "update.hpp"
//#include "cuStinger.hpp"

//#include "operators.cuh"

//#include "static_katz_centrality/katz.cuh"
//#include "streaming_katz_centrality/katz.cuh"


namespace custinger_alg {


katzCentralityDynamic::katzCentralityDynamic(custinger::cuStinger& custinger) :
                                       StaticAlgorithm(custinger),
                                       load_balacing(custinger),
									   hostKatzData.activeQueue(custinger)
									   {

    deviceKatzData = register_data(hostKatzData);
	memReleased = true;
	cout << "Oded remember to take care of memory de-allocation"  << endl;
	cout << "Oded need to figure out correct API for dynamic graph algorithms" << endl;
	cout << "Dynamic katz centrality algorithm needs to get both the original graph and the inverted graph for directed graphs" << endl;
}

katzCentralityDynamic::~katzCentralityDynamic() {
	release();
}


void katzCentralityDynamic::setInitParametersUndirected(int32_t maxIteration_, int32_t K_,degree_t maxDegree_){
	kcStatic.setInitParameters(maxIteration_,K_,maxDegree_,false);
	isDirected=false;
}
void katzCentralityDynamic::setInitParametersDirected(int32_t maxIteration_, int32_t K_,degree_t maxDegree_, cuStinger* invertedGraph__){
	kcStatic.setInitParameters(maxIteration_,K_,maxDegree_,false);
	invertedGraph=invertedGraph__;
	isDirected=true;

}

void katzCentralityDynamic::init(){
	if(memReleased==false){
		release();
		memReleased=true;
	}
	// Initializing the static graph KatzCentrality data structure
	kcStatic.init();

	// deviceKatzData = (katzDataStreaming*)allocDeviceArray(1, sizeof(katzDataStreaming));
	gpu::allocate(deviceKatzData, 1);

	// copyArrayHostToHost((void*)kcStatic.getHostKatzData(),&hostKatzData, 1, sizeof(katzData));
	// copyArrayDeviceToDevice((void*)kcStatic.getDeviceKatzData(),deviceKatzData, 1, sizeof(katzData));
	gpu::copyHostToHost(kcStatic.getHostKatzData(), 1, &hostKatzData,);
	gpu::copyDeviceToDevice(kcStatic.getDeviceKatzData(),1,deviceKatzData);

	gpu::allocate(hostKatzData.newPathsCurr, hostKatzData.nv);
	gpu::allocate(hostKatzData.newPathsPrev, hostKatzData.nv);
	gpu::allocate(hostKatzData.active, hostKatzData.nv);
//	hostKatzData.newPathsCurr = (ulong_t*) allocDeviceArray((hostKatzData.nv), sizeof(ulong_t));
//	hostKatzData.newPathsPrev = (ulong_t*) allocDeviceArray((hostKatzData.nv), sizeof(ulong_t));
	// hostKatzData.active 	  = (int*)    allocDeviceArray((hostKatzData.nv), sizeof(int));

	// cusLB = new cusLoadBalance(custinger.nv+1);

	// hostKatzData.activeQueue.Init(custinger.nv+1);
	// hostKatzData.activeQueue.resetQueue();

	hostKatzData.activeQueue.clear();

	syncDeviceWithHost();
}


void katzCentralityDynamic::runStatic(){

	// Executing the static graph algorithm
	kcStatic.reset();
	kcStatic.run(custinger);

	// copyArrayHostToHost((void*)kcStatic.getHostKatzData(),&hostKatzData, 1, sizeof(katzData));
	// copyArrayDeviceToDevice((void*)kcStatic.getDeviceKatzData(),deviceKatzData, 1, sizeof(katzData));
	copyHostToHost(kcStatic.getHostKatzData(), 1, &hostKatzData,);
	gpu::copyDeviceToDevice(kcStatic.getDeviceKatzData(),1,deviceKatzData);

	hostKatzData.iterationStatic = hostKatzData.iteration;
	syncDeviceWithHost();

	// Initializing the fields of the dynamic graph algorithm
	// allVinG_TraverseVertices<katzCentralityDynamicOperator::initStreaming>(custinger,deviceKatzData);
	forAllVertices<katzCentralityDynamicOperator::initStreaming>(custinger,deviceKatzData);
}

void katzCentralityDynamic::release(){
	if(memReleased==true)
		return;
	memReleased=true;

	// delete cusLB;


	// hostKatzData.activeQueue.freeQueue();
	gpu::free(hostKatzData.newPathsCurr);
	gpu::free(hostKatzData.newPathsPrev);
	gpu::free(hostKatzData.active);
	gpu::free(deviceKatzData);
	kcStatic.release();
}


int32_t katzCentralityDynamic::getIterationCount(){
	syncHostWithDevice();
	return hostKatzData.iteration;
}

void katzCentralityDynamic::batchUpdateInsertion(BatchUpdate &bu){
	processUpdate(bu,true);
}

void katzCentralityDynamic::batchUpdateDeleted(BatchUpdate &bu){
	processUpdate(bu,false);
}


void katzCentralityDynamic::processUpdate(BatchUpdate &bu, bool isInsert){

	// Resetting the queue of the active vertices.
	hostKatzData.activeQueue.clear();
	hostKatzData.iteration = 1;

	syncDeviceWithHost();

	// Initialization of insertions or deletions is slightly different.
	if(isInsert){
		allEinA_TraverseEdges<katzCentralityDynamicOperator::setupInsertions>(custinger, deviceKatzData,bu);
	}else{
		allEinA_TraverseEdges<katzCentralityDynamicOperator::setupDeletions>(custinger, deviceKatzData,bu);
	}
	syncHostWithDevice();

	hostKatzData.iteration = 2;
	hostKatzData.nActive = hostKatzData.activeQueue.getQueueEnd();

	while(hostKatzData.iteration < hostKatzData.maxIteration && hostKatzData.iteration < hostKatzData.iterationStatic){
		hostKatzData.alphaI = pow(hostKatzData.alpha,hostKatzData.iteration);
		syncDeviceWithHost();

		allVinA_TraverseVertices<katzCentralityDynamicOperator::initActiveNewPaths>(custinger, deviceKatzData, hostKatzData.activeQueue.getQueue(), hostKatzData.nActive);

		// Undirected graphs and directed graphs need to be dealt with differently.
		if(!isDirected){
			forAllEdges(katzCentralityDynamicOperator::findNextActive)(hostKatzData.activeQueue, load_balacing );
			// allVinA_TraverseEdges_LB<katzCentralityDynamicOperator::findNextActive>(custinger,deviceKatzData, *cusLB,hostKatzData.activeQueue);
			syncHostWithDevice(); // Syncing queue info
			forAllEdges(katzCentralityDynamicOperator::updateActiveNewPaths)(hostKatzData.activeQueue, load_balacing );			
			// allVinA_TraverseEdges_LB<katzCentralityDynamicOperator::updateActiveNewPaths>(custinger,deviceKatzData, *cusLB,hostKatzData.activeQueue);
		}
		else{			
			allVinA_TraverseEdges_LB<katzCentralityDynamicOperator::findNextActive>(*invertedGraph,deviceKatzData, *cusLB,hostKatzData.activeQueue);
			syncHostWithDevice(); // Syncing queue info
			allVinA_TraverseEdges_LB<katzCentralityDynamicOperator::updateActiveNewPaths>(*invertedGraph,deviceKatzData, *cusLB,hostKatzData.activeQueue);
		}
		syncHostWithDevice(); // Syncing queue info

		// Checking if we are dealing with a batch of insertions or deletions.
		if(isInsert){
			allEinA_TraverseEdges<katzCentralityDynamicOperator::updateNewPathsBatchInsert>(custinger, deviceKatzData,bu);
		}else{
			allEinA_TraverseEdges<katzCentralityDynamicOperator::updateNewPathsBatchDelete>(custinger, deviceKatzData,bu);
		}
		syncHostWithDevice();

		hostKatzData.nActive = hostKatzData.activeQueue.getQueueEnd();
		syncDeviceWithHost();
		allVinA_TraverseVertices<katzCentralityDynamicOperator::updatePrevWithCurr>(custinger, deviceKatzData, hostKatzData.activeQueue.getQueue(), hostKatzData.nActive);
		syncHostWithDevice();

		hostKatzData.iteration++;

	}
	if(hostKatzData.iteration>2){
		syncDeviceWithHost();
		allVinA_TraverseVertices<katzCentralityDynamicOperator::updateLastIteration>(custinger, deviceKatzData, hostKatzData.activeQueue.getQueue(), hostKatzData.nActive);
		syncHostWithDevice();
	}
	// Resetting the fields of the dynamic graph algorithm for all the vertices that were active
	allVinA_TraverseVertices<katzCentralityDynamicOperator::initStreaming>(custinger, deviceKatzData, hostKatzData.activeQueue.getQueue(), hostKatzData.nActive);
	// Need to use the queue for this one.
	// forAllVertices<katzCentralityDynamicOperator::initStreaming>(custinger,deviceKatzData);
}


}// cuStingerAlgs namespace

