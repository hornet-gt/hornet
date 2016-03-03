
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <malloc.h>
#include <inttypes.h>

void readGraphDIMACS(char* filePath, int32_t** prmoff, int32_t** prmind, int32_t* prmnv, int32_t* prmne)
{
    FILE *fp = fopen (filePath, "r");
    int32_t nv, ne;

    char* line = NULL;

    // Read data from file
    int32_t temp, lineRead;
    size_t bytesRead = 0;
    getline (&line, &bytesRead, fp);

    sscanf (line, "%d %d", &nv, &ne);   

    free(line);

    // printf("nv =%u, ne=%u\n",nv,ne );
    int32_t * off = (int32_t *) malloc ((nv + 2) * sizeof (int32_t));
    int32_t * ind = (int32_t *) malloc ((ne * 2) * sizeof (int32_t));
    off[0] = 0;
    off[1] = 0;
    int32_t counter = 0;
    int32_t u;
    line = NULL;
    bytesRead = 0;


    for (u = 1; (temp = getline (&line, &bytesRead, fp)) != -1; u++)
    {
        int32_t neigh = 0;
        int32_t v = 0;
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


