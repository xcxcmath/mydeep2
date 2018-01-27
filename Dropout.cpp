#include "Dropout.h"

namespace mydeep {
    namespace layer {

        Dropout::Dropout(double alive_ratio) {
            assert(alive_ratio <= 1 &&
                   alive_ratio >= 0 &&
                   "Ratio must be between zero and one.");

            m_param[ParamKey::ratio] = Matrix::Constant(1, 1, alive_ratio);
        }

        Dropout::Dropout(const Param &param)
                : Hidden(param)
        {

        }

        Matrix Dropout::predict(const Matrix &x) {
            return x * m_param[ParamKey::ratio](0, 0);
        }

        Matrix Dropout::forward(const Matrix &x) {
            m_in = x;
            const auto r = m_param[ParamKey::ratio](0, 0);

            std::random_device rd;
            std::bernoulli_distribution dist(r);

            m_mask = x.unaryExpr([&rd, &dist](const double&){
                return static_cast<double>(dist(rd)); // DO NOT USE IMPLICIT CAST
            });

            return m_out = m_in.cwiseProduct(m_mask);
        }

        BackOutput Dropout::backward(const Matrix &delta) {
            m_backout.delta = delta.cwiseProduct(m_mask);
            return m_backout;
        }

    }
}