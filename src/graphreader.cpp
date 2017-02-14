#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <malloc.h>
#include <inttypes.h>
#include <unordered_map>
#include "cuStingerDefs.hpp"

using namespace std;

bool hasOption(const char* option, int argc, char **argv) {
  for (int i = 1; i < argc; i++) {
	  if (strcmp(argv[i], option) == 0)
		  return true;
  }
  return false;
}

/**
 * Merges two sorted arrays while removing duplicates. Writes result to output buffer,
 * and returns length of merged array.
 *
 * http://stackoverflow.com/a/13830837/1925767
 */
/// Merges two sorted arrays into one array while removing duplicate elements.
int mergeSortedRemoveDuplicates(int *out, int *a, int *b, long aLen, long bLen) {
	int i, j, k, temp;
	i = j = k = 0;
	while (i < aLen && j < bLen) {
		temp = a[i] < b[j] ? a[i++] : b[j++];
        for ( ; i < aLen && a[i] == temp; i++);
        for ( ; j < bLen && b[j] == temp; j++);
        out[k++] = temp;
	}
    while (i < aLen) {
        temp = a[i++];
        for ( ; i < aLen && a[i] == temp; i++);
        out[k++] = temp;
    }
    while (j < bLen) {
        temp = b[j++];
        for ( ; j < bLen && b[j] == temp; j++);
        out[k++] = temp;
    }
	return k;
}

/// Function for parsing "DIMACS 10 Graph Challenge" graphs.
void readGraphDIMACS(char* filePath, length_t** prmoff, vertexId_t** prmind, vertexId_t* prmnv, length_t* prmne, int isRmat)
{
    FILE *fp = fopen (filePath, "r");
    vertexId_t nv;
    length_t ne;

    char* line = NULL;

    // Read data from file
    int32_t temp, lineRead;
    size_t bytesRead = 0;
    getline (&line, &bytesRead, fp);

    sscanf (line, "%d %d", &nv, &ne);   

    free(line);

    length_t * off = (length_t *) malloc ((nv + 2) * sizeof (length_t)); 
    nv++;
    vertexId_t * ind;
    if(!isRmat){
        ind = (vertexId_t *) malloc ((ne * 2) * sizeof (vertexId_t));        
        ne *= 2;        
    }
    else{
        ind = (vertexId_t *) malloc ((ne) * sizeof (vertexId_t));        
    }
    

    off[0] = 0;
    off[1] = 0;
    length_t counter = 0;
    vertexId_t u;
    line = NULL;
    bytesRead = 0;

    for (u = 1; (temp = getline (&line, &bytesRead, fp)) != -1; u++)
    {
        vertexId_t neigh = 0;
        vertexId_t v = 0;
        char *ptr = line;
        int read = 0;
        char tempStr[1000];
        lineRead = 0;
        while (lineRead < bytesRead && (read = sscanf (ptr, "%s", tempStr)) > 0)
        {
            v = atoi(tempStr);
            read = strlen(tempStr);
            ptr += read + 1;
            lineRead = read + 1;
            neigh++;
            ind[counter++] = v;
        }
        off[u + 1] = off[u] + neigh;
        free(line);
        bytesRead = 0;
    }
    fclose (fp);

    *prmnv = nv;
    *prmne = ne;
    *prmind = ind;
    *prmoff = off;
}

int hostCompareIncrement (const void *a, const void *b){
  return (int) (*(int64_t *) a - *(int64_t *) b);
}




/// Function for parsing SNAP graphs.
void readGraphSNAP  (char* filePath, length_t** prmoff, vertexId_t** prmind, vertexId_t* prmnv, length_t* prmne, bool undirected){
    vertexId_t nv,*src,*dest,*ind;
    length_t   ne,*degreeCounter,*off;
    nv = ne = -1;

    printf("Warning: SNAP reader may relabel vertex IDs\n");

    const int MAX_CHARS = 100;
    char temp[MAX_CHARS];
    FILE *fp = fopen (filePath, "r");

    // scan for SNAP header
    while (nv == -1 || ne == -1) {
    	fgets(temp, MAX_CHARS, fp);
    	sscanf(temp, "# Nodes: %d Edges: %d\n", &nv,&ne);
    }
    if (undirected)
    	ne *= 2;

    src = (vertexId_t *) malloc ((ne ) * sizeof (vertexId_t));
    dest = (vertexId_t *) malloc ((ne ) * sizeof (vertexId_t));
    degreeCounter = (length_t*)malloc((nv+1) * sizeof(length_t));
    off = (length_t*)malloc((nv+1) * sizeof(length_t));
    ind = (vertexId_t*)malloc((ne) * sizeof(vertexId_t));

    vertexId_t counter=0;
    for(vertexId_t v=0; v<nv;v++)
        degreeCounter[v]=0;

    vertexId_t srctemp,desttemp;
    while(counter<ne)
    {
        fgets(temp, MAX_CHARS, fp);
        sscanf(temp, "%d %d\n", (vertexId_t*)&srctemp, (vertexId_t*)&desttemp);
        src[counter]=srctemp;
        dest[counter]=desttemp;
        counter++;
        if (undirected) {
            src[counter]=desttemp;
            dest[counter]=srctemp;
            counter++;
        }
    }
    fclose (fp);

    vertexId_t *src_sorted = (vertexId_t *) malloc (ne * sizeof (vertexId_t));
    vertexId_t *dest_sorted = (vertexId_t *) malloc (ne * sizeof (vertexId_t));
    memcpy(src_sorted, src, ne*sizeof(vertexId_t));
    memcpy(dest_sorted, dest, ne*sizeof(vertexId_t));
    qsort(src_sorted, ne, sizeof(vertexId_t), hostCompareIncrement);
    qsort(dest_sorted, ne, sizeof(vertexId_t), hostCompareIncrement);
    vertexId_t *vertices_sorted = (vertexId_t *) malloc (nv * sizeof(vertexId_t));
    int nGraphVertices = mergeSortedRemoveDuplicates(vertices_sorted, src_sorted, dest_sorted, ne, ne);
    free(src_sorted);
    free(dest_sorted);

    unordered_map<vertexId_t, vertexId_t> relabel_map;
    vertexId_t relabeledSrcId, relabeledDestId;
    for (length_t i=0; i<nv; i++) {
    	relabel_map[vertices_sorted[i]] = i;
    }

    for (int i=0; i<counter; i++) {
    	relabeledSrcId = relabel_map[src[i]];
    	degreeCounter[relabeledSrcId]++;
    }

    // build offsets array
    off[0]=0;
    for(vertexId_t v=0; v<nv;v++)
        off[v+1]=off[v]+degreeCounter[v];

    printf("Processed %d vertices, %d edges as input\n", nv, off[nv]);

    for(vertexId_t v=0; v<nv;v++)
        degreeCounter[v]=0;

    counter=0;
    while(counter<ne)
    {
    	relabeledSrcId = relabel_map[src[counter]];
    	relabeledDestId = relabel_map[dest[counter]];
        ind[off[relabeledSrcId]+degreeCounter[relabeledSrcId]++] = relabeledDestId;
        counter++;
    }

    free(src);
    free(dest);
    free(vertices_sorted);
    free(degreeCounter);
    *prmnv=nv;
    *prmne=ne;
    *prmind=ind;
    *prmoff=off;
}

/// Function for parsing Florida Matrix Market graphs.
void readGraphMatrixMarket(char* filePath, length_t** prmoff, vertexId_t** prmind, vertexId_t* prmnv, length_t* prmne, bool undirected){
    vertexId_t nv,*src,*dest,*ind;
    length_t   ne,*degreeCounter,*off;

    const int MAX_CHARS = 100;
    char temp[MAX_CHARS];
    FILE *fp = fopen(filePath, "r");

    while (fgets(temp, MAX_CHARS, fp) && *temp == '%'); // skip comments
    sscanf(temp, "%d %*s %d\n", &nv,&ne); // read Matrix Market header
    if (undirected)
    	ne *= 2;
    src = (vertexId_t *) malloc ((ne ) * sizeof (vertexId_t));
    dest = (vertexId_t *) malloc ((ne ) * sizeof (vertexId_t));
    degreeCounter = (length_t*)malloc((nv+1) * sizeof(length_t));
    off = (length_t*)malloc((nv+1) * sizeof(length_t));
    ind = (vertexId_t*)malloc((ne) * sizeof(vertexId_t));

    int64_t counter=0;
    for(int64_t v=0; v<nv;v++)  {
        degreeCounter[v]=0;
    }

    vertexId_t srctemp, desttemp;
    while(counter<ne)
    {
        fgets(temp, MAX_CHARS, fp);
        sscanf(temp, "%d %d %*s\n", (vertexId_t*)&srctemp, (vertexId_t*)&desttemp);
        src[counter]=srctemp-1;
        dest[counter]=desttemp-1;
        degreeCounter[srctemp-1]++;
        counter++;
        if (undirected) {
            src[counter]=desttemp-1;
            dest[counter]=srctemp-1;
            degreeCounter[desttemp-1]++;
            counter++;
        }
    }
    fclose (fp);

    off[0]=0;
    // fill offsets
    for(int v=0; v<nv;v++)
        off[v+1]=off[v]+degreeCounter[v];

    printf("Processed %d vertices, %d edges as input\n", nv, off[nv]);

    for(int v=0; v<nv;v++)
        degreeCounter[v]=0;

    counter=0;
    // fill adjacencies
    while(counter<ne)
    {
        ind[off[src[counter]]+degreeCounter[src[counter]]++]=dest[counter];
        counter++;
    }

    free(src);
    free(dest);
    free(degreeCounter);
    *prmnv=nv;
    *prmne=ne;
    *prmind=ind;
    *prmoff=off;
}


