
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <malloc.h>
#include <inttypes.h>

#include "cuStingerDefs.hpp"

void readGraphDIMACS(char* filePath, length_t** prmoff, vertexId_t** prmind, vertexId_t* prmnv, length_t* prmne)
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
    vertexId_t * ind = (vertexId_t *) malloc ((ne * 2) * sizeof (vertexId_t));
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

    nv++;
    ne *= 2;
    *prmnv = nv;
    *prmne = ne;
    *prmind = ind;
    *prmoff = off;
}


