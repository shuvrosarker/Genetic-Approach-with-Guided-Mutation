#include <stdio.h>
#include <string.h>



int parse_data(char *file_name, char *out_file)
{
    FILE *fp     = fopen(file_name, "r");
    FILE *fp_out = fopen(out_file, "w");
    
    if(fp == NULL) printf("Error Opening File ... %s\n", file_name);
    if(fp_out == NULL) printf("Error Opening File %s\n",out_file);
    
    int index = 0;
    float data[15];
    
    while( fscanf(fp, "%f %f %f %f %f %f %f %f %f %f %f %f %f %f", &data[0], &data[1], &data[2], &data[3],&data[4], &data[5], &data[6], &data[7], &data[8], &data[9], &data[10],&data[11], &data[12], &data[13]) != EOF)
    {
        fprintf(fp_out, "%d,", index++);
        for(int i = 0; i < 14; i++)
        {
            if(i < 13) fprintf(fp_out, "%0.3f,", data[i]);
            else fprintf(fp_out, "%0.3f\n", data[i]);
        }
    }
    
    fclose(fp);
    fclose(fp_out);
    
}

int main()
{
    parse_data((char *)"heart.data", (char *) "heart_train_data.txt");
    return 0;
}