#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <malloc.h>
#include <inttypes.h>
#include <assert.h>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include "cuStingerDefs.hpp"

using namespace std;

bool hasOption(const char* option, int argc, char **argv) {
  for (int i = 1; i < argc; i++) {
	  if (strcmp(argv[i], option) == 0)
		  return true;
  }
  return false;
}

bool sortByPairAsec(const std::pair<int,int> &a, const std::pair<int,int> &b) {
    if (a.first < b.first) {
      return true;
    } else if (a.first > b.first) {
      return false;
    } else {
      return (a.second < b.second);
    }
}

bool sortByPairDesc(const std::pair<int,int> &a, const std::pair<int,int> &b) {
    if (a.first > b.first) {
      return true;
    } else if (a.first < b.first) {
      return false;
    } else {
      return (a.second > b.second);
    }
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

/// Function for parsing SNAP graphs.
void readGraphSNAP  (char* filePath, length_t** prmoff, vertexId_t** prmind, vertexId_t* prmnv, length_t* prmne, bool undirected){
    printf("Warning: SNAP reader may relabel vertex IDs\n");

    vertexId_t nv = -1;
    length_t ne = -1;

    const int MAX_CHARS = 1000;
    char temp[MAX_CHARS];
    char *written;
    FILE *fp = fopen(filePath, "r");

    // scan for SNAP header comment
    written = fgets(temp, MAX_CHARS, fp);
    while ((nv == -1 || ne == -1) && written != NULL) {
        sscanf(temp, "# Nodes: %d Edges: %d\n", &nv,&ne);
            written = fgets(temp, MAX_CHARS, fp);
    }
    if ((nv == -1 || ne == -1) && written == NULL) {
        fprintf(stderr, "SNAP input file is missing header info\n");
        exit(-1);
    }
    while (written != NULL && *temp == '#') { // skip any other comments
        written = fgets(temp, MAX_CHARS, fp);
    }

    vector<pair<vertexId_t, vertexId_t> >  edges(ne);
    unordered_set<vertexId_t> vertex_set;
    vertexId_t counter=0;
    vertexId_t srctemp,desttemp;

    // read in edges
    while(counter<ne)
    {
        sscanf(temp, "%d %d\n", (vertexId_t*)&srctemp, (vertexId_t*)&desttemp);
        if (undirected) {
            edges[counter]= make_pair(min(srctemp, desttemp), max(srctemp, desttemp));
        } else {
            edges[counter] = make_pair(srctemp, desttemp);
        }
        vertex_set.insert(srctemp);
        vertex_set.insert(desttemp);
        counter++;
        fgets(temp, MAX_CHARS, fp);
    }
    fclose (fp);
    assert(vertex_set.size() == nv);
    printf("Original input: %d vertices, %lu edges\n", nv, edges.size());

    vector<pair<vertexId_t, vertexId_t> > edges_final;
    sort(edges.begin(), edges.end(), sortByPairAsec);
    if (undirected) {     // convert to undirected edges, and remove potential duplicates
        vertexId_t prev_first = -1;
        vertexId_t prev_second = -1;
        vertexId_t first, second;
        int duplicates = 0;
        for (vector<pair<vertexId_t, vertexId_t> >::iterator pair = edges.begin(); pair != edges.end(); pair++) {
            first = pair->first;
            second = pair->second;
            if (first == prev_first && second == prev_second) {
                duplicates += 1;
            } else {
                edges_final.push_back(*pair);
                edges_final.push_back(make_pair(second, first));
            }
            prev_first = first;
            prev_second = second;
        }
        ne = edges_final.size();
        printf("Removed %d duplicate edges in conversion to undirected\n", duplicates);
    } else {
        edges_final = edges;
    }

    // sort graph vertices and use to create relabeling map
    vector<vertexId_t> vertices(vertex_set.begin(), vertex_set.end());
    sort(vertices.begin(), vertices.end());
    unordered_map<vertexId_t, vertexId_t> relabel_map;
    for (length_t i=0; i<nv; i++) {
        relabel_map[vertices[i]] = i;
    }

    // convert to CSR
    length_t *degreeCounter = (length_t*)calloc((nv+1), sizeof(length_t));
    length_t *off = (length_t*)malloc((nv+1) * sizeof(length_t));
    vertexId_t *ind = (vertexId_t*)malloc((ne) * sizeof(vertexId_t));

    vertexId_t relabeledSrcId, relabeledDestId;
    for (int i=0; i<edges_final.size(); i++) {
        relabeledSrcId = relabel_map[edges_final[i].first];
        degreeCounter[relabeledSrcId]++;
    }

    // build offsets array
    off[0]=0;
    for(vertexId_t v=0; v<nv;v++)
        off[v+1]=off[v]+degreeCounter[v];

    printf("Processed %d vertices, %d edges\n", nv, off[nv]);

    for(vertexId_t v=0; v<nv;v++)
        degreeCounter[v]=0;

    for (int i=0; i<edges_final.size(); i++)
    {
        relabeledSrcId = relabel_map[edges_final[i].first];
        relabeledDestId = relabel_map[edges_final[i].second];
        ind[off[relabeledSrcId]+degreeCounter[relabeledSrcId]++] = relabeledDestId;
    }

    free(degreeCounter);
    *prmnv=nv;
    *prmne=ne;
    *prmind=ind;
    *prmoff=off;
}

/// Function for parsing Florida Matrix Market graphs.
void readGraphMatrixMarket(char* filePath, length_t** prmoff, vertexId_t** prmind, vertexId_t* prmnv, length_t* prmne, bool undirected){
    vertexId_t nv = -1;
    length_t ne = -1;
    vertexId_t *ind;
    length_t *degreeCounter,*off;

    const int MAX_CHARS = 100;
    char temp[MAX_CHARS];
    FILE *fp = fopen(filePath, "r");

    while (fgets(temp, MAX_CHARS, fp) && *temp == '%'); // skip comments
    sscanf(temp, "%d %*s %d\n", &nv,&ne); // read Matrix Market header

    vector<pair<vertexId_t, vertexId_t> >  edges(ne);
    int64_t counter=0;
    vertexId_t srctemp, desttemp;
    while(counter<ne)
    {
        fgets(temp, MAX_CHARS, fp);
        sscanf(temp, "%d %d %*s\n", (vertexId_t*)&srctemp, (vertexId_t*)&desttemp);
        if (undirected) {
            edges[counter]= make_pair(min(srctemp-1, desttemp-1), max(srctemp-1, desttemp-1));
        } else {
            edges[counter] = make_pair(srctemp-1, desttemp-1);
        }
        counter++;
    }
    fclose (fp);
    printf("Original input: %d vertices, %lu edges\n", nv, edges.size());

    vector<pair<vertexId_t, vertexId_t> > edges_final;
    sort(edges.begin(), edges.end(), sortByPairAsec);
    if (undirected) {     // convert to undirected edges, and remove potential duplicates
        vertexId_t prev_first = -1;
        vertexId_t prev_second = -1;
        vertexId_t first, second;
        int duplicates = 0;
        for (vector<pair<vertexId_t, vertexId_t> >::iterator pair = edges.begin(); pair != edges.end(); pair++) {
            first = pair->first;
            second = pair->second;
            if (first == prev_first && second == prev_second) {
                duplicates += 1;
            } else {
                edges_final.push_back(*pair);
                edges_final.push_back(make_pair(second, first));
            }
            prev_first = first;
            prev_second = second;
        }
        ne = edges_final.size();
        printf("Removed %d duplicate edges in conversion to undirected\n", duplicates);
    } else {
        edges_final = edges;
    }

    degreeCounter = (length_t*)calloc((nv+1), sizeof(length_t));
    off = (length_t*)malloc((nv+1) * sizeof(length_t));
    ind = (vertexId_t*)malloc((ne) * sizeof(vertexId_t));

    vertexId_t relabeledSrcId, relabeledDestId;
    for (int i=0; i<edges_final.size(); i++) {
        // printf("srcId: %d\n", edges_final[i].first);
        relabeledSrcId = edges_final[i].first;
        degreeCounter[relabeledSrcId]++;
    }

    off[0]=0;
    // fill offsets
    for(int v=0; v<nv;v++)
        off[v+1]=off[v]+degreeCounter[v];

    printf("Processed %d vertices, %d edges\n", nv, off[nv]);

    for(int v=0; v<nv;v++)
        degreeCounter[v]=0;

    // fill adjacencies
    for (int i=0; i<edges_final.size(); i++)
    {
        relabeledSrcId = edges_final[i].first;
        relabeledDestId = edges_final[i].second;
        ind[off[relabeledSrcId]+degreeCounter[relabeledSrcId]++] = relabeledDestId;
    }

    free(degreeCounter);
    *prmnv=nv;
    *prmne=ne;
    *prmind=ind;
    *prmoff=off;
}


