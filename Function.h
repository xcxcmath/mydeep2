#ifndef MYDEEP2_FUNCTIONS_H
#define MYDEEP2_FUNCTIONS_H

#include "Common.h"

namespace mydeep {
    namespace Function {
        namespace Activation {
            using Function = std::function<double(const double &)>;

            struct FuncPair {
                Function f;
                Function df;
            };

            extern const FuncPair ReLU;
            extern const FuncPair Sigmoid;
            extern const FuncPair Tanh;
        }

        namespace Output {
            using OutputFunction = std::function<Matrix(const Matrix&)>;
            using LossFunction = std::function<double(const Matrix&, const Matrix&)>;

            struct FuncPair {
                OutputFunction f;
                LossFunction loss;
            };

            extern const FuncPair Softmax;
            extern const FuncPair Identity;
        }
    }
}

#endif //MYDEEP2_FUNCTIONS_H
