#include <stdio.h>
#include <string.h>



int parse_data(char *file_name, char *out_file)
{
    FILE *fp     = fopen(file_name, "r");
    FILE *fp_out = fopen(out_file, "w");
    
    if(fp == NULL) printf("Error Opening File ... %s\n", file_name);
    if(fp_out == NULL) printf("Error Opening File %s\n",out_file);
    
    int index = 0;
    char line[400];
    
    while( fscanf(fp, "%s", line) != EOF )
    {
        if(line[strlen(line) - 1] == '2') line[strlen(line) - 1] = '1';
        else if(line[strlen(line) - 1] == '4') line[strlen(line) - 1] = '2';
        
        fprintf(fp_out, "%s\n", line);
    }
    
    fclose(fp);
    fclose(fp_out);
    
}

int main()
{
    parse_data((char *)"breast-cancer-wisconsin.data", (char *) "breast_cancer_train.txt");
    return 0;
}