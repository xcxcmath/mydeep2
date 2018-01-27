#include "Activation.h"

namespace mydeep {
    namespace layer {
        Activation::Activation(const Activation::FunctionPair &fp)
                :m_fp(fp)
        {

        }

        Activation::Activation(const Activation::Function &f, double h) {
            assert(h > 0 && "Parameter \"h\" must be over zero.");

            m_fp.f = f;
            m_fp.df = [f, h](const double &x){
                const double after = x+h, before = x-h;
                return (f(after) - f(before)) / (2.*h);
            };
        }

        Matrix Activation::predict(const Matrix &x) {
            return x.unaryExpr(m_fp.f);
        }

        BackOutput Activation::backward(const Matrix &delta) {
            m_backout.delta = delta.cwiseProduct(m_in.unaryExpr(m_fp.df));
            return m_backout;
        }

        const Activation::FunctionPair ReLU = {
                [](const double &x) -> double {
                    return x * (x>0);
                },
                [](const double &x) -> double {
                    return static_cast<double>(x>0);
                }
        };
        const Activation::FunctionPair Sigmoid = {
                [](const double &x) -> double {
                    return 1. / (1.+std::exp(-x));
                },
                [](const double &x) -> double {
                    const double exp = 1./(1.+std::exp(-x));
                    return exp * (1. - exp);
                }
        };
        const Activation::FunctionPair Tanh = {
                [](const double &x) -> double {
                    return std::tanh(x);
                },
                [](const double &x) -> double {
                    return std::pow(std::cosh(x), -2.);
                }
        };
    }
}