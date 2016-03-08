#include <fstream>
#include <iostream>
#include <vector>
#include <string>
#include <cstring>
#include <cmath>
#include <cstdlib>
#include <iomanip>
using namespace std;

//one single datapoint and it's information
struct datapoint{
    int dclass; //stores whether the datapoint is of classification 1 or 2
    vector<double> decfeature; //stores the datapoints coordinates/features
    datapoint(int d, const vector<double> &f)
        :dclass(d)
    {
        decfeature = f;
    }
};

void printtrace(vector<int> currfeat, double acc){
    cout << "\tUsing feature(s) {";
    for(int i = 0; i < currfeat.size(); ++i){
        if(i+1 == currfeat.size()){
            cout << currfeat[i]+1;
        }
        else{
            cout << currfeat[i]+1 << ",";
        }
    }
    cout << "} accuracy is " << acc*100 << "%" << endl;
}
void printbest(vector<int> currfeat, double acc){
    cout << "Feature set {";
    for(int i = 0; i < currfeat.size(); ++i){
        if(i+1 == currfeat.size()){
            cout << currfeat[i]+1;
        }
        else{
            cout << currfeat[i]+1 << ",";
        }
    }
    cout << "} was best, accuracy is " << acc*100 << "%" << endl;
}
void printresult(vector<int> currfeat, double acc){
    cout << "Finished search: Feature set {";
    for(int i = 0; i < currfeat.size(); ++i){
        if(i+1 == currfeat.size()){
            cout << currfeat[i]+1;
        }
        else{
            cout << currfeat[i]+1 << ",";
        }
    }
    cout << "} was best, with an accuracy of " << acc*100 << "%" << endl;
}
//takes in data from a 
void readInData(vector<datapoint> &datasheet, string fileName){
    string line;
    ifstream dataFile(fileName.c_str());
    if(dataFile){
        while(getline(dataFile, line)){
            vector<char*> features;
            vector<double> featuresdec;
            double basedec = 0;
            int pow10 = 0;
            
            char* temp = const_cast<char*> (line.c_str());
            char *token = strtok(temp, " ");
            char *pointclass = token;
            if(token != NULL){
                pointclass = token;
                token = strtok(NULL, " ");
            }
            
            while(token != NULL){
                features.push_back(token);
                token = strtok(NULL, " ");
            }
            
            for(auto &e: features){
                token = strtok(e, "e");
                if(token != NULL){
                    basedec = stod(token,nullptr);
                    token = strtok(NULL, "e");
                }
                if(token != NULL){
                    if(token[0] == '+'){
                        pow10 = atoi(token+1);
                    }
                    else if(token[0] == '-'){
                        pow10 = atoi(token+1) * -1;
                    }
                    token = strtok(NULL, "e");
                }
                featuresdec.emplace_back(basedec * pow(10.0, pow10));
            }
            
            datapoint point(atoi(pointclass),featuresdec);
            datasheet.emplace_back(point);
        }
        dataFile.close();
    }
    else{
        cerr << "File unable to open" << endl;
        exit(1);
    }
}

double avgmean(const vector<datapoint> &datasheet, int feature){
    double sum = 0.0;
    for(auto &e : datasheet){
        sum += e.decfeature[feature];
    }
    return sum / static_cast<double>(datasheet.size());
}

double stddev(const vector<datapoint> &datasheet, int feature, double mean){
    double sum = 0.0;
    for(auto &e : datasheet){
        double temp = e.decfeature[feature] - mean;
        sum += pow(temp, 2);
    }
    return sqrt(sum / static_cast<double>(datasheet.size() - 1.0));
}

void normalize(vector<datapoint> &datasheet){
    //normalizes one feature at a time
    for(int i = 0; i < datasheet[0].decfeature.size(); ++i){
        double mean = avgmean(datasheet, i);
        double stdev = stddev(datasheet,i,mean);
        for(auto &e : datasheet){
            e.decfeature[i] = (e.decfeature[i] - mean) / stdev;
        }
    }
}

double distance(vector<double> a,vector<double> b,const vector<int> &currfeats){
    double sum = 0.0;
    for(int i = 0; i < currfeats.size(); ++i){
        sum += pow((a[currfeats[i]] - b[currfeats[i]]), 2);
    }
    return sqrt(sum);
}

int nearestneighborclassif(const vector<datapoint> &datasheet, int testpoint,
    const vector<int> &currfeats){//currfeats tells which features to test
    //Store 3 nearest neighbors, these will vote on which class the testpoint should be
    double min1 =9999999999.0, min2 = 9999999999.0, min3= 9999999999.0; //stores the distance value
    int loc1 = -1, loc2 = -1, loc3 = -1;
    for(int i = 0; i < datasheet.size(); ++i){
        if(i != testpoint){
            double dist = distance(datasheet[i].decfeature,
                datasheet[testpoint].decfeature,currfeats);
                // cout << "dist is " << dist << endl;
            //determines if the training data point is closer than currently stored ones
            if(dist < min3){
                if(dist < min2){
                    if(dist < min1){
                        min3 = min2;
                        min2 = min1;
                        min1 = dist;
                        
                        loc3 = loc2;
                        loc2 = loc1;
                        loc1 = i;
                    }
                    else{
                        min3 = min2;
                        min2 = dist;
                        
                        loc3 = loc2;
                        loc2 = i;
                    }
                }
                else{
                    min3 = dist;
                    
                    loc3 = i;
                }
            }
        }
    }
    if(datasheet[loc1].dclass == datasheet[loc2].dclass 
        || datasheet[loc1].dclass == datasheet[loc3].dclass){
        return datasheet[loc1].dclass;
    }
    if(datasheet[loc2].dclass == datasheet[loc3].dclass){
        return datasheet[loc2].dclass;
    }
}

double findaccuracy(const vector<datapoint> &datasheet, const vector<int> &currfeats){
    int counter = 0;
    for(int i = 0; i < datasheet.size(); ++i){
        int temp = nearestneighborclassif(datasheet, i, currfeats);
        if(temp == datasheet[i].dclass){
            ++counter;
        }
    }
    //cout << "counter is: " << counter << endl << endl;
    return static_cast<double>(counter)/datasheet.size();
}

vector<int> forwardsearch(const vector<datapoint> &datasheet){
    cout << "Begginning Forwards Selection" << endl << endl;
    vector<int> bestfeats;//best feature list overall
    vector<int> currfeats;//best feature list of current options
    vector<int> stepfeats;//used to store a feature list for the current step
    double maxacc = 0.0;
    //goes through the entire search tree
    for(int i = 0; i < datasheet[0].decfeature.size(); ++i){
        double curracc = 0.0;
        stepfeats = currfeats;
        //searches each feature to see which one to add
        for(int j = 0; j < datasheet[0].decfeature.size(); ++j){
            vector<int> temp = stepfeats;
            bool repeat = false;
            //makes sure there are no repeats of features in the list
            for(int x = 0; x < currfeats.size(); ++x){
                if(j == currfeats[x]){
                    repeat = true;
                }
            }
            if(!repeat){
                temp.push_back(j);
                double accuracy = findaccuracy(datasheet, temp);
                printtrace(temp, accuracy);
                //stores max
                if(accuracy > curracc){
                    curracc = accuracy;
                    currfeats = temp;
                }
                if(accuracy > maxacc){
                    maxacc = accuracy;
                    bestfeats = temp;
                }
            }
        }
        cout << endl;
        if(maxacc != curracc){
            cout << "(Warning: Accuracy has decreased. Continuing search in case of local maxima)"
                << endl;
        }
        printbest(currfeats,curracc);
        cout << endl;
    }
    
    printresult(bestfeats,maxacc);
    return bestfeats;
}

vector<int> backwardelim(const vector<datapoint> &datasheet){
    cout << "Begginning Backwards Elimination" << endl << endl;
    vector<int> bestfeats;//best feature list overall
    vector<int> currfeats;//best feature list of current options
    vector<int> stepfeats;//used to store a feature list for the current step
    //fills current with all the features
    for(int init = 0; init < datasheet[0].decfeature.size(); ++init){
        currfeats.push_back(init);
    }
    bestfeats = currfeats;
    double maxacc = findaccuracy(datasheet, bestfeats);
    printtrace(bestfeats, maxacc);
    cout << endl;
    printbest(bestfeats,maxacc);
    cout << endl;
    for(int i = 0; i < datasheet[0].decfeature.size()-1; ++i){
        double curracc = 0.0;
        stepfeats = currfeats;
        //goes through the set to determine which one to delete
        for(int j = 0; j < stepfeats.size(); ++j){
            vector<int> temp = stepfeats;
            temp.erase(temp.begin()+j);
            double accuracy = findaccuracy(datasheet, temp);
            
            printtrace(temp, accuracy);
            if(accuracy > curracc){
                curracc = accuracy;
                currfeats = temp;
            }
            if(accuracy > maxacc){
                maxacc = accuracy;
                bestfeats = temp;
            }
        }
        cout << endl;
        if(maxacc != curracc){
            cout << "(Warning: Accuracy has decreased. Continuing search in case of local maxima)"
                << endl;
        }
        printbest(currfeats,curracc);
        cout << endl;
    }

    printresult(bestfeats,maxacc);
    return bestfeats;
}

vector<int> bidirectionalsearch(const vector<datapoint> &datasheet){
    //variables for performing the forward part of the search
    vector<int> forwardfeat, forwardcurr, forwardstep;
    double formaxacc = 0.0;
    
    //variables for performing the backwards part of the search
    vector<int> backfeat, backcurr, backstep;
    double backmaxacc = 0.0;
    
    //initializing the backwards array to have all features
    for(int init = 0; init < datasheet[0].decfeature.size(); ++init){
        backfeat.push_back(init);
    }
    backcurr = backfeat;
    backstep = backfeat;
    backmaxacc = findaccuracy(datasheet, backfeat);
    int i = 0;
    while(!(backcurr == forwardcurr)){
        //forwardsearch reinit
        double forcurracc = 0.0; //current max for j number of features
        forwardstep = forwardcurr;
        
        //backwards reinit
        double backcuracc = 0.0; //current max for j number of features
        backstep = backcurr;
        
        for(int j = 0; j < datasheet[0].decfeature.size(); ++j){
            //step forward search
            vector<int> fortemp = forwardstep;
            bool repeat = false;
            bool inback = false;//checks if this variable is in back
            for(int x = 0; x < forwardcurr.size(); ++x){
                if(j == forwardcurr[x]){
                    repeat = true;
                }
            }
            //because we must make sure that features added must be in back
            //we check the current feature we plan to add if it is in back
            for(auto &b : backcurr){
                if( j == b ){
                    inback = true;
                }
            }
            if(!repeat && inback){
                
                fortemp.push_back(j);
                double foraccuracy = findaccuracy(datasheet, fortemp);
                printtrace(fortemp, foraccuracy);
                if(foraccuracy > forcurracc){
                    forcurracc = foraccuracy;
                    forwardcurr = fortemp;
                }
                if(foraccuracy > formaxacc){
                    formaxacc = foraccuracy;
                    forwardfeat = fortemp;
                }
            }
        }
        cout << endl;
        if(formaxacc != forcurracc){
            cout << "(Warning: Accuracy has decreased. Continuing search in case of local maxima)"
                << endl;
        }
        printbest(forwardcurr, forcurracc);
        cout << endl;
        //step backwards elimination
        double backcurracc = 0.0;
        backstep = backcurr;
        
        for(int j = 0; j < backstep.size(); ++j){
            //making sure that the variable we are going to delete isn't in forward search
            
            bool inforward = false;
            for(auto &f : forwardcurr){
                if(f == backstep[j]){
                    inforward = true;
                }
            }
            if(!inforward){
                
                vector<int> backtemp = backstep;
                backtemp.erase(backtemp.begin()+j);
                double backaccuracy = findaccuracy(datasheet, backtemp);
                
                printtrace(backtemp, backaccuracy);
                if(backaccuracy > backcurracc){
                    backcurracc = backaccuracy;
                    backcurr = backtemp;
                }
                if(backaccuracy > backmaxacc){
                    backmaxacc = backaccuracy;
                    backfeat = backtemp;
                }
            }
        }
        cout << endl;
        if(backmaxacc != backcurracc){
            cout << "(Warning: Accuracy has decreased. Continuing search in case of local maxima)"
                << endl;
        }
        printbest(backcurr,backcurracc);
        cout << endl;
        
        ++i;
        int tempcount = 0;
        for(auto &a : backcurr){
            for(auto &c : forwardcurr){
                if(a == c){
                    ++tempcount;
                }
            }
        }
        if(backcurr.size() == tempcount && tempcount == forwardcurr.size()){
            break;
        }
    }
    
    printresult(forwardfeat,formaxacc);
    return forwardfeat;
    
    
}

int main(){
    vector<datapoint> datasheet;
    vector<int> results;
    string fileName;
    int searchsel = -1;
    cout << "Welcome to Features Selection Algorithm" << endl;
    cout << "Type in the name of the file to test: ";
    cin >> fileName;
    cout << endl;
    cout << "Importing data...";
    readInData(datasheet,fileName);
    cout << "Done" << endl;
    cout << "Normalizing data...";
    normalize(datasheet);
    cout << "Done" << endl;
        
    while(searchsel != 1 && searchsel != 2 && searchsel != 3){
        cout << "Type the number of the algorithm you want to run" << endl << endl;
        cout << "\t1) Forward Selection" << endl;
        cout << "\t2) Backwards Elimination" << endl;
        cout << "\t3) Original Algorithm: Bidirectional Search" << endl;
        cout << endl;
        cin >> searchsel;
        if(searchsel == 1){
            results = forwardsearch(datasheet);
        }
        else if(searchsel == 2){
            results = backwardelim(datasheet);
        }
        else if(searchsel == 3){
            results = bidirectionalsearch(datasheet);
        }
        else{
            cout << "Incorrect Selection: Please choose a valid option" << endl;
            cout << endl;
        }
    }
}