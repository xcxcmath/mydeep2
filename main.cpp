#include <iostream>
#include <ctime>

#include "Dense.h"

using namespace mydeep;
using namespace std;

using deepMatrix = mydeep::Matrix;

const string rsp[3] = {"r", "s", "p"};
constexpr int rsp_result[3] = {2, 0, 1};

int str2rsp(const string & str){
    for(unsigned i = 0 ; i < 3; ++i){
        if(rsp[i] == str) return i;
    }
    return -1;
}

int main() {
    cout.precision(10);

    network::Network network;

    network.insert(AFFINE(15));
    network.insert(BATCHNORM());
    network.insert(RELU);
    network.insert(AFFINE(15));
    network.insert(BATCHNORM());
    network.insert(RELU);
    network.insert(AFFINE(3));
    network.insert(SOFTMAX);

    optimizer::Adam sgd(&network, 0.01);

    random_device rd;
    uniform_int_distribution<int> dist(0, 2);
    constexpr int batch_size = 10;
    constexpr int batch_time = 10000;

    cout << "Deep Learning Sequence Start..\n\n";
    clock_t begin = clock();

    for(unsigned i = 0 ; i < batch_time ; ++i){
        deepMatrix input_mat = deepMatrix::Zero(3, batch_size);
        deepMatrix ans_mat = deepMatrix::Zero(3, batch_size);
        for(unsigned bidx = 0 ; bidx < batch_size ; ++bidx){
            int test_in = dist(rd);
            input_mat(test_in, bidx) = 1.;

            int test_ans = rsp_result[test_in];
            ans_mat(test_ans, bidx) = 1.;
        }

        double loss = sgd.learn(input_mat, ans_mat);
        if((i+1) % (batch_time / 10) == 0)cout << i+1 << " : " << loss << endl;
    }
    clock_t end = clock();
    cout << "\nDeep Learning Sequence Finished..\n";
    cout << "Elapsed time : " << ((double)(end-begin)/CLOCKS_PER_SEC)<<" sec\n\n";
/*
    while(1){
        cout << "input : ";
        string input_str; cin >> input_str;

        int input = str2rsp(input_str);
        if(input <0) break;

        deepMatrix input_mat = deepMatrix::Zero(3, 1);
        input_mat(input, 0) = 1.;
        deepMatrix output_mat = network.predict(input_mat);

        int r, c;
        output_mat.maxCoeff(&r, &c);

        cout << endl << " r : " << output_mat(0, 0) * 100. << " %" << endl;
        cout << " s : " << output_mat(1, 0) * 100. << " %" << endl;
        cout << " p : " << output_mat(2, 0) * 100. << " %" << endl;
        cout << " answer : " << rsp[r] << " ";
        cout << (str2rsp(rsp[r]) == rsp_result[input] ? "CORRECT!" : "WRONG ANSWER") << endl << endl;
    }
*/
    return 0;
}