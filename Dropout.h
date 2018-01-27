#ifndef MYDEEP2_DROPOUT_H
#define MYDEEP2_DROPOUT_H

#include "Hidden.h"

namespace mydeep {
    namespace layer {
        class Dropout : public Hidden {
        public:
            explicit Dropout(double alive_ratio = 0.5);
            explicit Dropout(const Param &param);

            Matrix predict(const Matrix &x) override;
            Matrix forward(const Matrix &x) override;
            BackOutput backward(const Matrix &delta) override;

        protected:
            Matrix m_mask;
        };
    }
}

#endif //MYDEEP2_DROPOUT_H
