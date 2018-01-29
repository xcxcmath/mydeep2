#ifndef MYDEEP2_BATCHNORM_H
#define MYDEEP2_BATCHNORM_H

#include "Hidden.h"

namespace mydeep {
    namespace layer {
        class BatchNorm : public Hidden {
        public:
            explicit BatchNorm(double momentum = 0.9,
                               double eps = 1e-7);
            explicit BatchNorm(const Param &param);

            Matrix predict(const Matrix &x) override;
            Matrix forward(const Matrix &x) override;
            BackOutput backward(const Matrix &delta) override;

        protected:
            Matrix m_mean;
            Matrix m_xc;
            Matrix m_var;
            Matrix m_norm;

            static const std::function<double(const double &)> pow_minus_half;
            static const std::function<double(const double &)> pow_minus_1_5;
        };
    }
}

#endif //MYDEEP2_BATCHNORM_H
