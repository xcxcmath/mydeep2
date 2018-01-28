#include "Adam.h"

namespace mydeep {
    namespace optimizer {

        Adam::Adam(network::Network *net,
                   double learning_rate,
                   double beta1,
                   double beta2,
                   double eps,
                   unsigned int time_max)
                : Optimizer(net, learning_rate)
        {
            m_hp[HyperParamKey::beta1] = beta1;
            m_hp[HyperParamKey::beta2] = beta2;
            m_hp[HyperParamKey::epsilon] = eps;
            m_hp[HyperParamKey::time] = 0.;
            m_hp[HyperParamKey::time_max] = time_max;
        }

        Adam::Adam(network::Network *net,
                   const HyperParam &hp,
                   const Avg &avg)
                : Optimizer(net, hp, avg) {

        }

        ParamVector Adam::get_update(const ParamVector &grad) {
            const auto &lr = m_hp[HyperParamKey::lr];
            const auto &b1 = m_hp[HyperParamKey::beta1];
            const auto &b2 = m_hp[HyperParamKey::beta2];
            const auto &eps = m_hp[HyperParamKey::epsilon];
            const auto &t_m = m_hp[HyperParamKey::time_max];
            auto &t = m_hp[HyperParamKey::time];

            double lr_t = lr * -1.;
            if(t < t_m){
                ++t;
                lr_t *= std::sqrt(1.-std::pow(b2, t)) / (1.-std::pow(b1, t));
            }

            ParamVector ret;

            const auto sz = grad.size();

            for(size_t i = 0; i < sz; ++i){
                Param here;
                if(t == 1.){
                    m_avg[AvgKey::first].emplace_back();
                    m_avg[AvgKey::second].emplace_back();
                }

                auto &am = m_avg[AvgKey::first];
                auto &av = m_avg[AvgKey::second];

                for(const auto &pair: grad[i]){
                    const auto &key = pair.first;
                    const auto &val = pair.second;
                    const auto val_sqr = val.array().pow(2).matrix();

                    if(t == 1.){
                        am[i][key] = av[i][key] = Matrix::Zero(val.rows(), val.cols());
                    }

                    auto &m_here = am[i][key];
                    auto &v_here = av[i][key];
                    m_here = m_here * b1 + val * (1. - b1);
                    v_here = v_here * b2 + val_sqr * (1. - b2);
                    here[key] = (m_here.array() / (v_here.array() + eps).pow(0.5) * lr_t).matrix();
                }

                ret.push_back(here);

            }
            return ret;
        }
    }
}