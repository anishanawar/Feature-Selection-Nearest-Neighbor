#include <iostream>
#include <vector>
#include <cmath>
#include "limits.h"
#include "instance.h"
#include "problem.h"

using namespace std;

Problem::Problem(vector<Instance> v)                                            //constructor
{
    dataset = v;
}

int Problem::dataset_size()                                                     //returns the size of the dataset
{
    return dataset.size();
}

double Euclidian(vector<double> test, vector<double> train, vector<int> index)  //calculates Euclidian distance
{
    double dist = 0.0;
    
    for(int i = 0; i < index.size(); ++i)
    {
        dist += pow(test.at(index.at(i)) - train.at(index.at(i)), 2);
    }
    return dist;    //removed sqrt
}

double Problem::Nearest_N(vector<int> v)                                        //takes in the feature sets and calculates the %
{
    double correct = 0; double tot_count = 0;
    
    for(int i = 0; i < dataset.size(); ++i)
    {
        double s_dist = INT_MAX;
        Instance nn;
        
        for(int j = 0; j < dataset.size(); ++j)
        {
            if(j == i)
            {
                continue;
            }
            
            double temp = Euclidian(dataset.at(i).get_features(), dataset.at(j).get_features(), v);
           
            if(temp < s_dist)                                                   //keeps track of the nearest neighbor
            {
                s_dist = temp;
                nn = dataset.at(j);
            }
        }
    
        if(nn.get_type() == dataset.at(i).get_type())                           //calculate the % correct/ total
        {
            correct += 1;
        }
        tot_count += 1;
    }
    
    return (correct/tot_count);
}
