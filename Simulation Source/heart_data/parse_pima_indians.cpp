#include <stdio.h>
#include <string.h>



int parse_data(char *file_name, char *out_file)
{
    FILE *fp     = fopen(file_name, "r");
    FILE *fp_out = fopen(out_file, "w");
    
    if(fp == NULL) printf("Error Opening File ... %s\n", file_name);
    if(fp_out == NULL) printf("Error Opening File %s\n",out_file);
    
    int index = 0;
    float data[9];
    
    while( fscanf(fp, "%f %f %f %f %f %f %f %f %f ", &data[0], &data[1], &data[2], &data[3],&data[4], &data[5], &data[6], &data[7], &data[8]) != EOF)
    {
        fprintf(fp_out, "%d,", index++);
        for(int i = 0; i < 9; i++)
        {
            if(i < 8) fprintf(fp_out, "%0.3f,", data[i]);
            else fprintf(fp_out, "%0.3f\n", data[i]+1);
        }
    }
    
    fclose(fp);
    fclose(fp_out);
    
}

int main()
{
    parse_data((char *)"australian.data.dat", (char *) "australian_data.txt");
    return 0;
}