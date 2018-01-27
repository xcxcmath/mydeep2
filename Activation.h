#ifndef MYDEEP2_ACTIVATION_H
#define MYDEEP2_ACTIVATION_H

#include "Hidden.h"
#include "Function.h"

namespace mydeep {
    namespace layer {
        class Activation : public Hidden {
        public:
            using Function = std::function<double(const double &)>;
            struct FunctionPair {
                Function f;
                Function df;
            };

            explicit Activation(const FunctionPair &fp);
            explicit Activation(const Function &f,
                                double h = 1e-4);

            Matrix predict(const Matrix &x) override;
            BackOutput backward(const Matrix &delta) override;

        protected:
            FunctionPair m_fp;
        };

        extern const Activation::FunctionPair ReLU;
        extern const Activation::FunctionPair Sigmoid;
        extern const Activation::FunctionPair Tanh;
    }
}

#endif //MYDEEP2_ACTIVATION_H
