#ifndef MYDEEP2_OUTPUT_H
#define MYDEEP2_OUTPUT_H

#include "Layer.h"

namespace mydeep {
    namespace layer {
        class Output : public Layer {
        public:
            using OutputFunction = std::function<Matrix(const Matrix&)>;
            using LossFunction = std::function<double(const Matrix&, const Matrix&)>;

            struct FunctionPair {
                OutputFunction f;
                LossFunction loss;
            };

            explicit Output(const FunctionPair &fp);

            Matrix predict(const Matrix &x) override;
            double forward(const Matrix &x, const Matrix &ans);
            Matrix backward(const Matrix &ans);

        protected:
            FunctionPair m_fp;
        };

        extern const Output::FunctionPair Softmax;
        extern const Output::FunctionPair Identity;
    }
}

#endif //MYDEEP2_OUTPUT_H
