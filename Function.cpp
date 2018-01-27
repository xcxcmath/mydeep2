#include "Function.h"

namespace mydeep {
    namespace Function {
        namespace Activation {
            const FuncPair ReLU = {
                    [](const double &x) -> double {return x * (x>0);},
                    [](const double &x) -> double {return (x>0);}
            };
            const FuncPair Sigmoid = {
                    [](const double &x) -> double {return 1./(1.+std::exp(-x));},
                    [](const double &x) -> double {
                        const auto exp = std::exp(-x);
                        return exp / (std::pow(1.+exp, 2.));
                    }
            };
            const FuncPair Tanh {
                    [](const double &x) -> double {return std::tanh(x);},
                    [](const double &x) -> double {return std::pow(std::cosh(x), -2.);}
            };
        }

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