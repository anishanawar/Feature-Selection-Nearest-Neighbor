#include <iostream>
#include "instance.h"

using namespace std;

Instance::Instance()
{
    type = 0;
}

void Instance::append_feature(double f)
{
    features.push_back(f);
    return;
}

void Instance::set_type(double i)                   //setter fimcton for the type of an instance
{
    type = i;
    return;
}

void Instance::print()                              //prints the instances (mostly for testing)
{
    cout << "type: " << type << endl;
    
    for(int i = 0; i < features.size(); ++i)
    {
        cout << features.at(i) << " ";
    }cout << endl;
    
    return;
}

double Instance::get_type()                         //returns the type of the instance
{
    return type;
}

vector<double> Instance::get_features()             //returns the features of the instance
{
    return features;
}