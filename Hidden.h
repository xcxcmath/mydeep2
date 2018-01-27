#ifndef MYDEEP2_HIDDEN_H
#define MYDEEP2_HIDDEN_H

#include "Layer.h"

namespace mydeep {
    namespace layer {
        enum class ParamKey {
            initialized,
            weight, bias,   //Affine
            init_stddev,
            ratio,          //Dropout
            beta, gamma,    //Batch Norm
            mean, var,
            momentum_epsilon,
        };
        using Param = std::map<ParamKey, Matrix>;

        struct BackOutput {
            Matrix delta;
            Param gradient;
        };

        class Hidden : public Layer {
        public:
            explicit Hidden();
            explicit Hidden(const Param &param);

            Matrix predict(const Matrix &x) override = 0;
            virtual Matrix forward(const Matrix &x);
            virtual BackOutput backward(const Matrix &delta)= 0;

            void update(const Param &param);
            Param param() const;

        protected:
            Param m_param;
            BackOutput m_backout;
        };
    }
}

#endif //MYDEEP2_HIDDEN_H
