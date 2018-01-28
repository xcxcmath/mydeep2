#include "NAG.h"

namespace mydeep {
    namespace optimizer {

        NAG::NAG(network::Network *net,
                 double learning_rate,
                 double momentum)
                : Momentum(net, learning_rate, momentum)
        {

        }

        NAG::NAG(network::Network *net,
                 const HyperParam &hp,
                 const Avg &avg)
                : Momentum(net, hp, avg)
        {

        }

        ParamVector NAG::get_update(const ParamVector &grad) {
            const auto sz = grad.size();

            if(m_hp[HyperParamKey::time] == 0.){
                auto ret = get_gradient_step(grad);
                m_avg[AvgKey::first] = ret;
                auto &momentum = m_avg[AvgKey::first];
                for(size_t i = 0; i < sz; ++i)
                    for(const auto &pair: momentum[i])
                        momentum[i][pair.first] *= -1.;

                m_hp[HyperParamKey::time] = 1.;
                return ret;
            }

            auto &momentum = m_avg[AvgKey::first];
            auto momentum_step = get_momentum_step(grad);
            for(size_t i = 0; i < sz; ++i)
                for(const auto &pair: grad[i])
                    momentum_step[i][pair.first] *= -1.;

            m_net->update(momentum_step);
            auto lg = m_net->flow(m_in, m_ans);
            auto grad_step = get_gradient_step(lg.gradient);
            auto ret = momentum;

            for(size_t i = 0; i < sz; ++i){
                for(const auto &pair: momentum[i]){
                    const auto &key = pair.first;

                    momentum[i][key] = (momentum_step[i][key] + grad_step[i][key])*-1.;

                    ret[i][key] = momentum[i][key] * -1.;
                }
            }

            return ret;
        }
    }
}