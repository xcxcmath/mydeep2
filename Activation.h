#ifndef MYDEEP2_ACTIVATION_H
#define MYDEEP2_ACTIVATION_H

#include "Hidden.h"

namespace mydeep {
    namespace layer {
        class Activation : public Hidden {
        public:
            using Function = std::function<double(const double &)>;
            struct FunctionPair {
                Function f;
                Function df;
            };
            static const FunctionPair ReLU;
            static const FunctionPair Sigmoid;
            static const FunctionPair Tanh;

            explicit Activation(const FunctionPair &fp);
            explicit Activation(const Function &f,
                                double h = 1e-4);

            Matrix predict(const Matrix &x) override;
            BackOutput backward(const Matrix &delta) override;

        protected:
            FunctionPair m_fp;
        };
    }
}

#endif //MYDEEP2_ACTIVATION_H
