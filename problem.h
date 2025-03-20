#include <iostream>
#include <vector>
#include "instance.h"

using namespace std;

class Problem
{
    private:
        vector<Instance> dataset;           //vector that holds the dataset
        
    public: 
        Problem(vector<Instance>);          //constructor
        int dataset_size();
        double Nearest_N(vector<int>);      //returns % based on features used
};