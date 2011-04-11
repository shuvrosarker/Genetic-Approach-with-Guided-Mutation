#include <stdio.h>
#include <string.h>


int main()
{
    char input[100];
    int count = 0;
    
    FILE *fp  = fopen("parsed_data2.txt","r");
    //FILE *fp2 = fopen("8_bit_data.txt","w");
    
    if(fp == NULL)
    {
        printf("Error Opening File\n");
        return -1;
    }
    
    while(fscanf(fp, "%s", input) != EOF)
    {
        printf("%d,%s\n", ++count, input);
    }
    
    fclose(fp);
    return 0;
}