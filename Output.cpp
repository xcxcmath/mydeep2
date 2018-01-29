#include "Output.h"

namespace mydeep {
    namespace layer {

        const Output::FunctionPair Output::Softmax = {
                [](const Matrix &x) -> Matrix {
                    const auto exp = (x.rowwise() - x.colwise().maxCoeff()).unaryExpr(cwise_exp());
                    const auto exp_sum = exp.colwise().sum().unaryExpr(cwise_pow(-1.)).row(0).asDiagonal();
                    return exp * exp_sum;
                },
                [](const Matrix &y, const Matrix &ans) -> double {
                    return (ans.cwiseProduct(y.unaryExpr(cwise_log()))).sum()
                           *-1. / static_cast<double>(y.cols());
                }
        };
        const Output::FunctionPair Output::Identity = {
                [](const Matrix &x) -> Matrix {
                    return x;
                },
                [](const Matrix &y, const Matrix &ans) -> double {
                    return (y-ans).unaryExpr(
                            [](const double &i) -> double {return i*i;}
                    ).sum()
                           /2./static_cast<double>(y.cols());
                }
        };

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


    }
}