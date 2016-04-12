
#ifndef _CU_STINGER_DEFS_INCLUDE_H_
#define _CU_STINGER_DEFS_INCLUDE_H_


typedef int8_t vtype_t;
typedef int8_t etype_t;
typedef int32_t vweight_t;
typedef int32_t eweight_t;
typedef int32_t vertexId_t;
typedef int32_t length_t;
typedef int32_t timestamp_t;

void* allocHostArray(length_t elements,int32_t eleSize);
void* allocDeviceArray(length_t elements,int32_t eleSize);
void freeHostArray(void* array);
void freeDeviceArray(void* array);

void copyArrayHostToHost    (void* hostSrc,  void* hostDst, length_t elements, int32_t eleSize);
void copyArrayHostToDevice  (void* hostSrc,  void* devDst,  length_t elements, int32_t eleSize);
void copyArrayDeviceToHost  (void* devSrc,   void* hostDst, length_t elements, int32_t eleSize);
void copyArrayDeviceToDevice(void* devSrc,   void* devtDst, length_t elements, int32_t eleSize);


#endif

