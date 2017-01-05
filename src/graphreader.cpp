#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <malloc.h>
#include <inttypes.h>

#include "cuStingerDefs.hpp"

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


void readGraphSNAP  (char* filePath, length_t** prmoff, vertexId_t** prmind, vertexId_t* prmnv, length_t* prmne){
    vertexId_t nv,*src,*dest,*ind;
    length_t   ne,*degreeCounter,*off;
    
    FILE *fp = fopen (filePath, "r");
    fscanf(fp, "# Nodes: %d Edges: %d\n", &nv,&ne);

    // printf ("Edge list reading: %d, %d\n",nv,ne);

    src = (vertexId_t *) malloc ((ne ) * sizeof (vertexId_t));    
    dest = (vertexId_t *) malloc ((ne ) * sizeof (vertexId_t));   
    degreeCounter = (length_t*)malloc((nv+1) * sizeof(length_t)); 
    off = (length_t*)malloc((nv+1) * sizeof(length_t));
    ind = (vertexId_t*)malloc((ne) * sizeof(vertexId_t));

    int64_t counter=0;
    char line[2000];size_t len=2000; char read;      

    for(int64_t v=0; v<nv;v++)  {
        degreeCounter[v]=0;
    }   
    
    while(counter<ne)
    {
        int64_t srctemp,desttemp;
        fscanf(fp, "%ld %ld\n", &srctemp,&desttemp);

        src[counter]=srctemp;
        dest[counter]=desttemp;
        degreeCounter[srctemp]++;

        // This is needed for some faulty SNAP graphs
        // src[counter]=srctemp-1;
        // dest[counter]=desttemp-1;
        // degreeCounter[srctemp-1]++;
        counter++;
    }
    fclose (fp);

    off[0]=0;
    // completes offset array
    for(int v=0; v<nv;v++)  {
        off[v+1]=off[v]+degreeCounter[v];
    }

    for(int v=0; v<nv;v++){
        degreeCounter[v]=0;
    }
    counter=0;
    while(counter<ne)
    {
        ind[off[src[counter]]+degreeCounter[src[counter]]++]=dest[counter];
            
        counter++;
    } 
    
    // if(0) {
    //   for (int i = 0; i < (nv); i++)
    //     {
    //       qsort (&ind[off[i]], off[i + 1] - off[i], sizeof (int64_t),
    //          hostCompareIncrement);
    //     }
    // }
    
    free(src);
    free(dest);
    free(degreeCounter);    
    *prmnv=nv;
    *prmne=ne;
    *prmind=ind;
    *prmoff=off;
}

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

    int64_t srctemp, desttemp;
    while(counter<ne)
    {
        int64_t srctemp,desttemp;
        fgets(temp, MAX_CHARS, fp);
        sscanf(temp, "%ld %ld %*s\n", (long*)&srctemp, (long*)&desttemp);
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
    for(int v=0; v<nv;v++)  {
        off[v+1]=off[v]+degreeCounter[v];
    }

    for(int v=0; v<nv;v++){
        degreeCounter[v]=0;
    }

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


