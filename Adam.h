#ifndef MYDEEP2_ADAM_H
#define MYDEEP2_ADAM_H

#include "Optimizer.h"

namespace mydeep {
    namespace optimizer {
        class Adam : public Optimizer {
        public:
            explicit Adam(network::Network *net,
                          double learning_rate = 0.001,
                          double beta1 = 0.9,
                          double beta2 = 0.999,
                          double eps = 1e-8,
                          unsigned time_max = 10000u);

            explicit Adam(network::Network *net,
                          const HyperParam &hp,
                          const Avg &avg);

        protected:
            ParamVector get_update(const ParamVector &grad) override;
        };
    }
}

#endif //MYDEEP2_ADAM_H
