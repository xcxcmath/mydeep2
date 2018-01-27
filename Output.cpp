#include "Output.h"

namespace mydeep {
    namespace layer {

        Output::Output(const Output::FunctionPair &fp)
                :m_fp(fp)
        {

        }

        Matrix Output::predict(const Matrix &x) {
            return m_fp.f(x);
        }

        double Output::forward(const Matrix &x, const Matrix &ans) {
            m_in = x;
            m_out = predict(m_in);
            return m_fp.loss(m_out, ans);
        }

        Matrix Output::backward(const Matrix &ans) {
            return (m_out - ans) / static_cast<double>(m_out.cols());
        }

        const Output::FunctionPair Softmax = {
                [](const Matrix &x) -> Matrix {
                    const auto exp = (x.rowwise() - x.colwise().maxCoeff()).array().exp();
                    return (exp.rowwise() / exp.colwise().sum()).matrix();
                },
                [](const Matrix &y, const Matrix &ans) -> double { //Operator error can be ignored
                    return (ans.array() * y.array().log()).sum()
                           *-1. / static_cast<double>(y.cols());
                }
        };
        const Output::FunctionPair Identity = {
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