#include "f2n2.h"
#include <iostream>
#include <vector>
#include <set>
using namespace std;

//This defination will change with each data set
#define TRAIN_SIZE 384
#define ATTR_NUM 8
#define NUM_CLASS 2
#define NUM_CLASS_MARCHAND 2
#define LMAX 200
#define MAX_NAME_SIZE 50

class Marchand :  public nn
{
    private:
        
	int num_validation;
	int monotone_increament;
	
	float inputs[TRAIN_SIZE][ATTR_NUM];
	float outputs[TRAIN_SIZE][NUM_CLASS];
	float validation_input[TRAIN_SIZE][ATTR_NUM];
	float validation_output[TRAIN_SIZE][NUM_CLASS];
	
	float error_limit;
	int num_inputs, max_epoch;
	vector< set< vector<float> > >classes;
	vector< int > positive_hidden_indexes;
	vector< int > negative_hidden_indexes;
	vector< float >:: const_iterator temp;
	char file_name[MAX_NAME_SIZE];
	float time_to_stop; // in seconds, initial value Inf
	
    public:
	
	Marchand(int num_in, int num_out, int num_inputs_a, int max_epoch_a, float min_error, int lmax, char *f_name, float stop_time);
	Marchand();
	int init_marchand(int num_in, int num_out, int num_inputs_a, int max_epoch_a, float min_error, int lmax, char *f_name, float stop_time);
	void doDecode( vector< set< vector<float> > > data);  // test function for encoding
	float execute(float *input, float *output);
	float calculate_average_error(int num_inputs);
	bool set_output_check(set< vector<float> > set_data, int output);
	int number_right_classification(set< vector<float> > set_data, int output);
	int set_hidden_to_output_weight();
	~Marchand();
	
	int set_monotone_increament(int n);
	int take_validation_data(float *v_input, float *v_output, int num_v_data); // This should be called before the execute function from main, num_v_data-> number of validation inputs
	float calculate_averafe_erro(int arg_num_outputs);
    
};