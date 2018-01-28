#ifndef MYDEEP2_OPTIMIZER_H
#define MYDEEP2_OPTIMIZER_H

#include "Network.h"

namespace mydeep {
    namespace optimizer {
        using Param = layer::Param;
        using ParamVector = network::Network::ParamVector;

        enum class HyperParamKey {
            lr, gamma, beta1, beta2, epsilon, time, time_max,
        };
        using HyperParam = std::map<HyperParamKey, double>;

        enum class AvgKey {
            first, second,
        };

        class Optimizer {
        public:
            using Avg = std::map<AvgKey, ParamVector>;

            explicit Optimizer(network::Network *net, double learning_rate = 0.01);
            explicit Optimizer(network::Network *net,
                               const HyperParam &hp,
                               const Avg &avg);

            double learn(const Matrix &x, const Matrix &ans);

            HyperParam hyper_param() const;
            Avg average() const;

        protected:
            virtual ParamVector get_update(const ParamVector &grad);
            ParamVector get_gradient_step(const ParamVector &grad);

            network::Network *m_net;

            HyperParam m_hp;
            Avg m_avg;

            Matrix m_in;
            Matrix m_ans;
        };
    }
}

#endif //MYDEEP2_OPTIMIZER_H
