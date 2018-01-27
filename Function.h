#ifndef MYDEEP2_FUNCTIONS_H
#define MYDEEP2_FUNCTIONS_H

#include "Common.h"

namespace mydeep {
    namespace function {

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
