#include <iostream>
#include <numeric>
#include <stdlib.h>
#include <inttypes.h>

#include "stx/btree_multiset.h"
#include "stx/btree_multimap.h"
#include "stx/btree_map.h"

#include "main.hpp"

using namespace std;

const int numberElements=1000000;


struct memVertexData{
	memVertexData(){}
	memVertexData(vertexId_t vertex_,uint64_t offset_,uint64_t size_){
		vertex=vertex_;
		offset=offset_;
		size=size_;
	}

	vertexId_t 	 vertex;
	uint64_t offset;
	uint64_t size;
};

typedef stx::btree_multimap<vertexId_t, memVertexData,std::less<vertexId_t> > offsetTree;

class edgeBlock
{
public:
	edgeBlock(){
		// cout <<"Hello1" << endl;
	}
	edgeBlock(uint64_t blockSize_){
		// cout <<"Hello2" << endl;
		blockSize=blockSize_;
		utilization=0;
		nextAvailable=0;
		//ODED - allocate cuda memory array.

		offTree = new offsetTree();
	}
	edgeBlock(const edgeBlock& eb){
		// cout <<"Hello3" << endl;
		*this=eb;
		// this->offTree.swap(eb.offTree);

	}
	~edgeBlock(){
	}

	void releaseInnerTree(){
		offTree->clear();
		delete offTree;
	}

	edgeBlock& operator=(const edgeBlock &eb ){
		this->blockSize=eb.blockSize;
		this->utilization=eb.utilization;
		this->nextAvailable=eb.nextAvailable;
		this->offTree = eb.offTree;
		return *this;
	}
	uint64_t addMemBlock(uint64_t memSize,vertexId_t v){
		uint64_t retval = nextAvailable;
		utilization+=memSize;
		nextAvailable+=memSize;

		memVertexData mvd(v,retval,memSize);
		offTree->insert2(v,mvd);
		return retval;
	}
	void removeMemBlock(vertexId_t v){
		offsetTree::iterator oi = offTree->find(v);
		if (oi!=offTree->end()){
			utilization-=oi.data().size;
			offTree->erase(oi);
		}
	}	
	uint64_t getAvailableSpace(){
		return blockSize-nextAvailable;
	}

	uint8_t* getMemoryBlockPtr(){
		return memoryBlockPtr;
	}

	edgeBlock* getEdgeBlockPtr(){
		return this;
	}

	void printPtr(){
		printf("%p, ",this);
		// cout << this << ", ";
	}

	uint64_t elementsInEdgeBlock(){
		return offTree->size();
	}

// private:
	uint8_t* memoryBlockPtr;
	uint64_t blockSize;
	uint64_t utilization;
	uint64_t availbility;
	uint64_t nextAvailable;
	offsetTree* offTree;
};

// typedef stx::btree_multimap<uint64_t, edgeBlock, std::less<uint64_t> > ebBPtree;

typedef edgeBlock* edgeBlockPtr;
typedef stx::btree_multimap<uint64_t, edgeBlockPtr, std::less<uint64_t> > ebBPtree;

struct memAllocInfo{
	memAllocInfo(){}
	memAllocInfo(edgeBlock* eb_,uint8_t*  ptr_){
		eb=eb_;
		ptr=ptr_;
	}
	edgeBlock* eb;
	uint8_t*  ptr;

};

class memoryManager
{
public:
	memoryManager(uint64_t blockSize_){
		blockSize=blockSize_;
		edgeBlock* temp = new edgeBlock(blockSize);
        btree.insert2(blockSize, temp);
	}

	~memoryManager(){
		for(ebBPtree::iterator bi=btree.begin(); bi != btree.end(); bi++){
			bi.data()->releaseInnerTree();
		}
		btree.clear();
	}

	memAllocInfo allocateMemoryBlock(uint64_t memSize,vertexId_t v){
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
	    		// cout << "Place found in the first element" << endl;
	    		edgeBlock* tempeb = bi.data();
	    		btree.erase(--bi);
	    		pos = tempeb->addMemBlock(memSize,v);
				btree.insert2(tempeb->getAvailableSpace(), tempeb);
				return memAllocInfo(tempeb->getEdgeBlockPtr(),tempeb->getMemoryBlockPtr()+pos);
	    }else if (bi==btree.end()){
	    	bi--;
		    if(bi.key() < memSize){
		    	// cout << "no space in tree" << endl;
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
				// printf("%p   %p \n",bi.data()->getEdgeBlockPtr(), tempeb->getEdgeBlockPtr());	    		
    		btree.erase(bi);
    		pos = tempeb->addMemBlock(memSize,v);
			btree.insert2(tempeb->getAvailableSpace(), tempeb);
			return memAllocInfo(tempeb->getEdgeBlockPtr(),tempeb->getMemoryBlockPtr()+pos);
	    }
	}

	uint8_t* removeMemoryBlock(edgeBlock* eb,vertexId_t v){
		eb->removeMemBlock(v);
	}

private:
public:
	ebBPtree btree;
	uint64_t blockSize;
};



#ifdef BTREE_DEBUG 
#define PRINT_TREE(tree) tree.print(std::cout);
#else 
#define PRINT_TREE(tree) 
#endif




int main(int argc, char const *argv[])
{

    if(false){
	    memoryManager mm(1000);

    }
    else{
    	memoryManager mm(100000000);
		memAllocInfo* maiArray = new memAllocInfo[numberElements];
	    srand(34234235);
    	for(int i=0; i<numberElements; i++){
        	maiArray[i] = mm.allocateMemoryBlock(rand() % 1000,i);

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
    }

	return 0;

}
