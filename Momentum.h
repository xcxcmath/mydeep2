#ifndef MYDEEP2_MOMENTUM_H
#define MYDEEP2_MOMENTUM_H

#include "Optimizer.h"

namespace mydeep {
    namespace optimizer {
        class Momentum : public Optimizer {
        public:
            explicit Momentum(network::Network *net,
                              double learning_rate = 0.01,
                              double momentum = 0.9);
            explicit Momentum(network::Network *net,
                              const HyperParam &hp,
                              const Avg &avg);

        protected:
            ParamVector get_update(const ParamVector &grad) override;

            ParamVector get_momentum_step(const ParamVector &grad);
        };
    }
}

#endif //MYDEEP2_MOMENTUM_H
