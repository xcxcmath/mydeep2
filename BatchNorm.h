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
        };
    }
}

#endif //MYDEEP2_BATCHNORM_H
