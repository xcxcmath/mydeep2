#ifndef MYDEEP2_NAG_H
#define MYDEEP2_NAG_H

#include "Momentum.h"

namespace mydeep {
    namespace optimizer {
        class NAG : public Momentum {
        public:
            explicit NAG(network::Network *net,
                         double learning_rate = 0.01,
                         double momentum = 0.9);
            explicit NAG(network::Network *net,
                         const HyperParam &hp,
                         const Avg &avg);

        protected:
            ParamVector get_update(const ParamVector &grad) override;
        };
    }
}

#endif //MYDEEP2_NAG_H
