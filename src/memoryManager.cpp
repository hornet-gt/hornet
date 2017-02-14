#include <iostream>
#include <numeric>
#include <stdlib.h>
#include <inttypes.h>


#include "memoryManager.hpp"


/// Constructing an edge block on the device.
edgeBlock::edgeBlock(uint64_t blockSize_){
	blockSize=blockSize_;
	utilization=0;
	nextAvailable=0;
	memoryBlockPtr = (uint8_t*)allocDeviceArray(blockSize,sizeof(uint8_t));
	// printf("%p\n",memoryBlockPtr);
	offTree = new offsetTree();
}

/// Copy constructor.
edgeBlock::edgeBlock(const edgeBlock& eb){
	*this=eb;
}

/// Free the inner tree in the  
void edgeBlock::releaseInnerTree(){
	offTree->clear();
	delete offTree;
	// printf("%p, ",memoryBlockPtr);
	freeDeviceArray(memoryBlockPtr);
}

edgeBlock& edgeBlock::operator=(const edgeBlock &eb ){
	this->blockSize=eb.blockSize;
	this->utilization=eb.utilization;
	this->nextAvailable=eb.nextAvailable;
	this->offTree = eb.offTree;
	return *this;
}
uint64_t edgeBlock::addMemBlock(uint64_t memSize,vertexId_t v){
	uint64_t retval = nextAvailable;
	utilization+=memSize;
	nextAvailable+=memSize;

	memVertexData mvd(v,retval,memSize);
	offTree->insert2(v,mvd);
	return retval;
}
void edgeBlock::removeMemBlock(vertexId_t v){
	offsetTree::iterator oi = offTree->find(v);
	if (oi!=offTree->end()){
		utilization-=oi.data().size;
		offTree->erase(oi);
	}
}	
uint64_t edgeBlock::getAvailableSpace(){
	return blockSize-nextAvailable;
}

uint8_t* edgeBlock::getMemoryBlockPtr(){
	return memoryBlockPtr;
}

edgeBlock* edgeBlock::getEdgeBlockPtr(){
	return this;
}

uint64_t edgeBlock::elementsInEdgeBlock(){
	return offTree->size();
}




//-----------------------------
//-----------------------------
//-----------------------------
//-----------------------------





memoryManager::memoryManager(uint64_t blockSize_){
	blockSize=blockSize_;
	edgeBlock* temp = new edgeBlock(blockSize);
    btree.insert2(blockSize, temp);
}

memoryManager::~memoryManager(){
	// cout << endl;
	// int count=0;
	for(ebBPtree::iterator bi=btree.begin(); bi != btree.end(); bi++){
		bi.data()->releaseInnerTree();
		delete bi.data();
		// cout << count++ << ", ";
	}
	// cout << endl;

	btree.clear();
}

memAllocInfo memoryManager::allocateMemoryBlock(uint64_t memSize,vertexId_t v){
	uint64_t pos;
	// cout << "Getting memory block of size : " << memSize << endl;
	if(btree.empty()){
		// cout << "Empty" << endl;
    	uint64_t totalMemSize=blockSize;
		if(memSize>blockSize){
			uint64_t numBlocks = memSize/blockSize + ceil(float(memSize%blockSize)/float(blockSize));
			totalMemSize = numBlocks*blockSize;
		}
		edgeBlock* tempeb = new edgeBlock(totalMemSize);
		pos = tempeb->addMemBlock(memSize,v);
		btree.insert2(tempeb->getAvailableSpace(), tempeb);
		return memAllocInfo(tempeb->getEdgeBlockPtr(),tempeb->getMemoryBlockPtr()+pos);
		// return tempeb.getEdgeBlockPtr()+pos;
	}
    ebBPtree::iterator bi = btree.upper_bound(memSize);

    if(bi==btree.begin()){
    		edgeBlock* tempeb = bi.data();
    		btree.erase(--bi);
    		pos = tempeb->addMemBlock(memSize,v);
			btree.insert2(tempeb->getAvailableSpace(), tempeb);
			return memAllocInfo(tempeb->getEdgeBlockPtr(),tempeb->getMemoryBlockPtr()+pos);
    }else if (bi==btree.end()){
    	bi--;
	    if(bi.key() < memSize){
	    	uint64_t totalMemSize=blockSize;
			if(memSize>blockSize){
				uint64_t numBlocks = memSize/blockSize + ceil(float(memSize%blockSize)/float(blockSize));
				totalMemSize = numBlocks*blockSize;
			}
			edgeBlock* tempeb = new edgeBlock(totalMemSize);
    		pos = tempeb->addMemBlock(memSize,v);
			btree.insert2(tempeb->getAvailableSpace(), tempeb);
			// cout << tempeb->getThisPtr() << endl;
			return memAllocInfo(tempeb->getEdgeBlockPtr(),tempeb->getMemoryBlockPtr()+pos);
	    }
	    else{
    		// cout << "Found some space" << endl;
    		edgeBlock* tempeb = bi.data();
    		btree.erase(bi);
    		pos = tempeb->addMemBlock(memSize,v);
			btree.insert2(tempeb->getAvailableSpace(), tempeb);				
			return memAllocInfo(tempeb->getEdgeBlockPtr(),tempeb->getMemoryBlockPtr()+pos);
	    }
    }
    else{
		edgeBlock* tempeb = bi.data();
		btree.erase(bi);
		pos = tempeb->addMemBlock(memSize,v);
		btree.insert2(tempeb->getAvailableSpace(), tempeb);
		return memAllocInfo(tempeb->getEdgeBlockPtr(),tempeb->getMemoryBlockPtr()+pos);
    }
    // cout << "something unexpected happened "<< endl;
}

void memoryManager::removeMemoryBlock(edgeBlock* eb,vertexId_t v){
	eb->removeMemBlock(v);
}

uint64_t memoryManager::getTreeSize(){
	return (uint64_t)btree.size();
}


#if(MEM_STAND_ALONE) 

const int numberElements=1000000;



int main(int argc, char const *argv[])
{
	memoryManager mm(2<<21);
	memAllocInfo* maiArray = new memAllocInfo[numberElements];
    srand(34234235);
	for(int i=0; i<numberElements; i++){
    	maiArray[i] = mm.allocateMemoryBlock(rand() % 10,i);

    	if (i>=numberElements-10 || i*2==numberElements){
			for(ebBPtree::iterator bi=mm.btree.begin(); bi != mm.btree.end(); bi++)
				cout << bi.data()->elementsInEdgeBlock() << ", ";
			cout << endl;
		}
	}
	for(ebBPtree::iterator bi=mm.btree.begin(); bi != mm.btree.end(); bi++)
		cout << bi.data()->elementsInEdgeBlock() << ", ";
	cout << endl;
	cout << "The size of tree is : " << mm.btree.size() << endl;

    srand(34234235);
	for(int i=0; i<numberElements; i++){
		// printf("%p %d ",maiArray[i].eb,i);
		fflush(stdout);

    	mm.removeMemoryBlock(maiArray[i].eb,i);

    	if (i>=numberElements-10 || i*2==numberElements){
			for(ebBPtree::iterator bi=mm.btree.begin(); bi != mm.btree.end(); bi++)
				cout << bi.data()->elementsInEdgeBlock() << ", ";
    		// cout << endl << maiArray[i].eb << " " << i << endl << flush;
			cout << endl;
		}
	}
	for(ebBPtree::iterator bi=mm.btree.begin(); bi != mm.btree.end(); bi++)
		cout << bi.data()->elementsInEdgeBlock() << ", ";
	cout << endl;

	cout << "The size of tree is : " << mm.btree.size() << endl;

	delete maiArray;

	return 0;

}

#endif

