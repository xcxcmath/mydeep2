#include "Function.h"

namespace mydeep {
    namespace function {
        namespace Output {
            const FuncPair Softmax = {
                    [](const Matrix &x) -> Matrix {
                        const auto exp = (x.rowwise() - x.colwise().maxCoeff()).array().exp();
                        return (exp.rowwise() / exp.colwise().sum()).matrix();
                    },
                    [](const Matrix &y, const Matrix &ans) -> double { //Operator error can be ignored
                        return (ans.array() * y.array().log()).sum()
                               *-1. / static_cast<double>(y.cols());
                    }
            };
            const FuncPair Identity = {
                    [](const Matrix &x) -> Matrix {
                        return x;
                    },
                    [](const Matrix &y, const Matrix &ans) -> double {
                        return (y-ans).array().pow(2.).sum()
                                /2./static_cast<double>(y.cols());
                    }
            };
        }
    }
}