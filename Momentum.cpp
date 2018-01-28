#include "Momentum.h"

namespace mydeep {
    namespace optimizer {

        Momentum::Momentum(network::Network *net, double learning_rate, double momentum)
                : Optimizer(net, learning_rate)
        {
            m_hp[HyperParamKey::gamma] = momentum;
            m_hp[HyperParamKey::time] = 0.;
        }

        Momentum::Momentum(network::Network *net, const HyperParam &hp, const Avg &avg)
                : Optimizer(net, hp, avg)
        {

        }

        ParamVector Momentum::get_update(const ParamVector &grad) {
            auto ret = get_gradient_step(grad);

            auto &momentum = m_avg[AvgKey::first];

            const auto sz = grad.size();

            if(m_hp[HyperParamKey::time] == 0.){
                momentum = ret;
                for(size_t i = 0; i < sz; ++i)
                    for(const auto &pair: ret[i])
                        momentum[i][pair.first] *= -1.;

                m_hp[HyperParamKey::time] = 1.;
                return ret;
            }

            auto momentum_step = get_momentum_step(grad);

            for(size_t i = 0; i < sz; ++i)
                for(const auto &pair: momentum[i]){
                    const auto &key = pair.first;
                    momentum[i][key] = momentum_step[i][key] - ret[i][key];
                    ret[i][key] = momentum[i][key] * -1.;
                }

            return ret;
        }

        ParamVector Momentum::get_momentum_step(const ParamVector &grad) {
            auto ret = m_avg[AvgKey::first];

            const auto sz = grad.size();

            for(size_t i = 0; i < sz; ++i)
                for(const auto &pair: ret[i])
                    ret[i][pair.first] *= m_hp[HyperParamKey::gamma];

            return ret;
        }

    }
}