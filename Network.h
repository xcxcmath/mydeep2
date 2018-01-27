#ifndef MYDEEP2_NETWORK_H
#define MYDEEP2_NETWORK_H

#include "Hidden.h"
#include "Output.h"

namespace mydeep {
    namespace network {
        class Network {
        public:
            using ParamVector = std::vector<layer::Param>;
            struct LossGradient {
                double loss;
                ParamVector gradient;
            };

            explicit Network();
            ~Network();

            Matrix predict(const Matrix &x);
            LossGradient flow(const Matrix &x, const Matrix &ans);
            void update(const ParamVector &pv);

            void insert(layer::Hidden *layer);
            void insert(layer::Output *layer);

        protected:
            std::vector<layer::Hidden*> m_layers;
            layer::Output *m_out;
        };
    }
}

#endif //MYDEEP2_NETWORK_H
