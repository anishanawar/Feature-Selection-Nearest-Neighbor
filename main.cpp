#include <iostream>      
#include <sstream>       
#include <fstream>       
#include <vector>        
#include <queue>         
#include <algorithm>     
#include <iterator>      
#include "limits.h"      
#include "instance.h"    
#include "problem.h"     

using namespace std;

// Struct: Feature_Set
// Holds a set of features (represented as indices) and its corresponding classification accuracy (computed using a nearest neighbor classifier).
// Also provides helper functions to adjust indices for printing and to compare feature sets based on accuracy.

struct Feature_Set
{
    double accuracy;           // Accuracy of the nearest neighbor classifier for this feature set
    vector<int> feat_in;       // Vector of selected feature indices (stored as 0-indexed; add 1 for printing)

    // Overloaded operator < used by priority_queue.
    // This allows the priority queue to sort Feature_Set objects based on their accuracy.
    const bool operator <(const Feature_Set& rhs) const
    {
      // Returns true if the current object's accuracy is less than that of rhs.
      return accuracy < rhs.accuracy;
    }
    
    // Function: inc_print increments each feature index by 1 for printing purposes.
    // Input: vector<int> b – a vector containing feature indices.
    // Returns: A new vector with each index incremented by 1.
    const vector<int> inc_print(vector<int> b) const
    {
        vector<int> v = b;    // Copy the input vector
        // Loop through all feature indices and add 1 to each (to convert from 0-indexed to 1-indexed)
        for(int i = 0; i < v.size(); ++i)
        {
            v.at(i) += 1;
        }
        return v;             // Return the adjusted vector
    }
    
    // Function: print
    // Purpose: Prints the feature set (with indices converted to 1-indexed) and its accuracy.
    const void print() const
    {
        // Get the adjusted (1-indexed) feature list
        vector<int> v = inc_print(feat_in);
        ostringstream oss;    // String stream for constructing the output
        
        // If there are features, copy all but the last one with a comma separator,
        // then add the last feature without a trailing comma.
        if (!v.empty())
        {
            copy(v.begin(), v.end()-1, ostream_iterator<int>(oss, ","));
            oss << v.back();
        }
        // Print the formatted feature set and its accuracy
        cout << "features set {" << oss.str() << "} with accuracy: " << accuracy << endl;
    }
};


// Function: parse
// Reads in a data file and converts each row into an Instance object.
// Each row is expected to have a type value followed by feature values.
// Input: string input_file – the name (or path) of the input file.
// Returns: A vector of Instance objects representing the dataset.

vector<Instance> parse(string input_file)
{
    ifstream infile;                        // Create an input file stream object
    infile.open(input_file.c_str());        // Open the file using its C-string representation
    
    vector<Instance> v;                     // Vector to store all Instance objects (the dataset)
    
    // Continue reading until end-of-file is reached
    while(infile.good())
    {
        Instance inst;                      // Create a new Instance for the current row
        double type = INT_MAX;              // Temporary variable to hold the type/class value
        vector<double> features;            // (Unused temporary vector; could be used for intermediate storage)
        double temp;                        // Temporary variable to hold each feature value
        
        string row;                         // String to hold the entire line
        getline(infile, row);               // Read one line from the file
        
        istringstream parse(row);           // Create a string stream for parsing the line
        
        parse >> type;                      // Extract the type (first value in the line)
        if(type == INT_MAX)                 // If type equals INT_MAX, assume end-of-data marker and break
        {
            break;
        }
        
        inst.set_type(type);                // Set the instance's type (class label)
        
        // Read the remaining values in the line as features and add them to the instance
        while(parse >> temp)
        {
            inst.append_feature(temp);      // Append each feature to the instance
        }
        
        v.push_back(inst);                  // Add the instance to the dataset vector
    }
    return v;                               // Return the complete dataset
}

// Function: is_in
// Purpose: Checks whether a given index exists in a vector of indices.
// Input: int index – the index to search for
//        vector<int> v – the vector of indices to search within
// Returns: true if index is found in the vector; false otherwise.

bool is_in(int index, vector<int> v)
{
    // Iterate through the vector elements
    for(int i = 0; i < v.size(); ++i)
    {
        if(index == v.at(i))
        {
            return true;  // Index found, return true
        }
    }
    return false;         // Index not found, return false
}

// Function: Forward
// Purpose: Performs forward selection for feature selection. Starting from an empty set, it iteratively adds features that improve classification accuracy using the Nearest Neighbor algorithm provided by the Problem object.
// Input: Problem prob – the problem instance containing the dataset and NN method int size – total number of features in the dataset

void Forward(Problem prob, int size)
{
    // Print initial dataset details (number of features and instances)
    cout << "This dataset has " << size << " features with "
         << prob.dataset_size() << " Instances:" << endl << endl;
         
    priority_queue<Feature_Set> pri_que;    // Priority queue to store candidate feature sets based on accuracy
    
    Feature_Set max;                        // Holds the feature set with the highest observed accuracy
    max.accuracy = 0;                       // Initialize maximum accuracy to 0
    Feature_Set temp;                       // Temporary Feature_Set object for current iteration
    vector<int> y;                          // Vector holding the current best set of feature indices (initially empty)
    
    // Evaluate and print the starting state (empty feature set)
    temp.feat_in = y;
    temp.accuracy = prob.Nearest_N(y);
    temp.print();
    
    bool warn = true;                       // Flag to control when a warning message is printed
    
    // Outer loop: each iteration attempts to add one feature
    for(int j = 0; j < size; ++j)
    {
        // Inner loop: test adding each feature not already in the current set
        for(int i = 0; i < size; ++i)
        {
            Feature_Set set;              // Create a candidate Feature_Set for the current trial
            vector<int> sel_features = y;   // Start with the current best feature set
            
            if(is_in(i, sel_features) == true)  // Skip if the feature is already selected
            {
                continue;
            }
            sel_features.push_back(i);      // Add the candidate feature
            set.feat_in = sel_features;       // Set the candidate feature set
            set.accuracy = prob.Nearest_N(sel_features);  // Evaluate accuracy with the new set
            pri_que.push(set);              // Push the candidate onto the priority queue
        }
        // Retrieve the candidate feature set with the highest accuracy
        temp = pri_que.top();
        
        // Update the best overall feature set if the candidate's accuracy is higher
        if(temp.accuracy > max.accuracy)
        {
            max = temp;
        }
        // If this is the first time the candidate accuracy drops, print a warning message.
        if(warn && temp.accuracy < max.accuracy)
        {
            warn = false;
            cout << endl << "Warning: accuracy decreasing, continuing search..." << endl << endl;
        }
        temp.print();                      // Print the best candidate for the current iteration
        y = temp.feat_in;                  // Update the current feature set to this candidate
        
        // Clear the priority queue to prepare for the next iteration
        while(!pri_que.empty())
        {
            pri_que.pop();
        }
    }
    // Print the best feature subset found after completing the forward selection
    cout << endl << "the best feature subset is ";
    max.print();
}

// Function: Backward
// Purpose: Performs backward elimination for feature selection. Starting from the full set of features, it iteratively removes features that yield the best accuracy, as measured by the Nearest Neighbor algorithm.
// Input: Problem prob – the problem instance containing the dataset and NN method int size – total number of features in the dataset

void Backward(Problem prob, int size)
{
    // Print initial dataset details
    cout << "This dataset has " << size << " features with "
         << prob.dataset_size() << " Instances:" << endl << endl;
         
    priority_queue<Feature_Set> pri_que;    // Priority queue to store candidate feature sets
    
    Feature_Set max;                        // Holds the best feature set observed during elimination
    max.accuracy = 0;                       // Initialize maximum accuracy to 0
    Feature_Set temp;                       // Temporary Feature_Set object for the current candidate
    vector<int> y;                          // Vector holding the current set of feature indices
    
    // Initialize y with all features (complete feature set)
    for(int i = 0; i < size; ++i)
    {
        y.push_back(i);
    }
    temp.feat_in = y;                       // Set the starting feature set to all features
    temp.accuracy = prob.Nearest_N(y);      // Evaluate accuracy using all features
    temp.print();                           // Print the starting state
    
    bool warn = true;                       // Flag to control when a warning is printed
    
    // Outer loop: each iteration attempts to remove one feature
    for(int j = 0; j < size; ++j)
    {
        // Inner loop: test removal of each feature in the current set
        for(int i = 0; i < size; ++i)
        {
            Feature_Set set;              // Create a candidate Feature_Set for removal
            vector<int> sel_features = y;   // Start with the current complete feature set
            
            if(is_in(i, sel_features) == false)  // Only consider features that are present in the set
            {
                continue;
            }
            // Remove feature i from the candidate set
            sel_features.erase(remove(sel_features.begin(), sel_features.end(), i), sel_features.end());
            // Note: The commented out line is not needed as the feature is removed.
            //sel_features.push_back(i);
            set.feat_in = sel_features;    // Assign the candidate feature set
            set.accuracy = prob.Nearest_N(sel_features);  // Evaluate accuracy without the feature
            pri_que.push(set);             // Push the candidate set onto the priority queue
        }
        // Retrieve the candidate with the highest accuracy after removal
        temp = pri_que.top();
        // Update the best overall feature set if this candidate's accuracy is higher
        if(temp.accuracy > max.accuracy)
        {
            max = temp;
        }
        // Print a warning if accuracy has decreased (only the first time)
        if(warn && temp.accuracy < max.accuracy)
        {
            warn = false;
            cout << endl << "Warning: accuracy decreasing, continuing search..." << endl << endl;
        }
        temp.print();                      // Print the candidate feature set for this iteration
        y = temp.feat_in;                  // Update the current feature set to the candidate
        
        // Clear the priority queue for the next iteration
        while(!pri_que.empty())
        {
            pri_que.pop();
        }
    }
    // Print the best feature subset found after completing backward elimination
    cout << endl << "the best feature subset is ";
    max.print();
}

// Function: main
// Purpose: Entry point of the program. It validates the command-line arguments, reads the input data file to construct the dataset, creates a Problem object, and then prompts the user to select a feature selection algorithm.

int main(int argc, char* argv[])
{
    // Ensure exactly two command-line arguments are provided:
    // the program name and the input file name.
    if(argc != 2)
    {
        cout << "Error: Invalid program call" << endl;
        return 1;
    }
    // The first argument (argv[0]) is the program name (unused here),
    // and the second argument (argv[1]) is the input file name.
    string file_name(argv[0]);
    string input_file(argv[1]);
    
    // Parse the input file to generate a vector of Instance objects (the dataset)
    vector<Instance> v = parse(input_file);
    
    // Create a Problem object using the loaded dataset
    Problem prob(v);
    // Determine the number of features from the first instance in the dataset
    int size = v.at(0).get_features().size();
    
    int input;
    // Print welcome message and algorithm selection options
    cout << "Welcome to Anisha's Feature Selection Algorithm" << endl;
    cout << "Type the number of the selection algorithm you want" << endl;
    cout << "\t 1) Forward Selection" << endl;
    cout << "\t 2) Backward Elimination" << endl;
    cout << "\t 3) Anisha's Special Algorithm" << endl;
    
    // Read the user's choice from standard input
    cin >> input;

    // Call the corresponding feature selection function based on the input
    if(input == 1)
    {
        Forward(prob, size);
    }
    else if(input == 2)
    {
        Backward(prob, size);
    }
    else
    {
        // If an unsupported option is chosen, print a default message.
        cout << "Anishas" << endl;
    }
    
    return 0;  // End program execution successfully.
}
