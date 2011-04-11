#include "f2n2.h"
#include <string.h>
#include <math.h>
#include <vector>
#include <algorithm>
#include <cstdio>
#include <iostream>


using namespace std;

#define TRAIN_SIZE 192 //number of test data
#define ATTR_NUM 8
#define NUM_CLASS 2
#define MAX_NODES 200

float output[TRAIN_SIZE][NUM_CLASS];
float input[TRAIN_SIZE][ATTR_NUM];

int read_data_from_file(char *file_name)
{
	int loop_i, loop_j;
	FILE *fp  = fopen(file_name,"r");
	//FILE *fp2 = fopen("output_file", "w");
	char comma;
	float temp;
	
	int num_row = 0;
	while( true )
	{
		if( fscanf(fp, "%f", &temp) == EOF) break; // the index is scanned here
		
		for(loop_i = 0; loop_i < ATTR_NUM ; loop_i++) fscanf(fp, "%c %f", &comma, &input[num_row][loop_i]);
		
                int value;
		fscanf(fp, "%c %d", &comma, &value);
		
		
		for( loop_j = 0; loop_j < NUM_CLASS; loop_j++ )
		{
			if(loop_j == (value- 1 ) )
			{
				output[num_row][loop_j] = 1;
			}
			else output[num_row][loop_j] = 0;
		}
		num_row++;
	}
		
	fclose(fp);
	//fclose(fp2);
	return 0;
    }

float get_error(char *file_name, char *input_file)
{

    read_data_from_file(input_file);
    
    float error = 0.0;
    float learning_rate = (double)(rand() % 10 + 2) / (double)(100);
    float momentum      = (double)(rand() % 7 + 1) / (double)(10);
		
    
    nn test_network;
    int layer[3] = {ATTR_NUM, 1, NUM_CLASS};
			
    test_network.create_nn(3, layer, 0,  learning_rate, momentum, MAX_NODES);
    test_network.load(file_name);
    
    printf("%s hidden neurons : %d\n", file_name, test_network.get_l_neurons(1));
			
    for( int i = 0; i < TRAIN_SIZE; i++)
    {
	test_network.calculate(input[i]);			
	error += test_network.get_error(output[i]);
    }
			
    printf("Total error: %f Average error: %f\n", error, (float)(error/ TRAIN_SIZE));
    return (float)(error/ TRAIN_SIZE);

}


int main()
{
    char input_file[50], net_file[50];
    strcpy(input_file, (char*) "pima_diabetes/pima_diabetes_test.txt");

    sprintf(net_file, "dnc_output.txt"); 
    printf("dnc error : %f\n", get_error(net_file, input_file) );
   
    sprintf(net_file, "marchand_output.txt"); 
    printf("marchand error : %f\n", get_error(net_file, input_file) );
   
    sprintf(net_file, "genetic_ouptut.txt"); 
    printf("genetic error : %f\n", get_error(net_file, input_file) );
    
    return 0;
}