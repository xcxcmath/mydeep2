#include "Optimizer.h"

namespace mydeep {
    namespace optimizer {

        Optimizer::Optimizer(network::Network *net, double learning_rate)
                :m_net(net)
        {
            m_hp[HyperParamKey::lr] = learning_rate;
        }

        Optimizer::Optimizer(network::Network *net,
                             const Optimizer::HyperParam &hp,
                             const Optimizer::Avg &avg)
                :m_net(net),
                 m_hp(hp),
                 m_avg(avg)
        {

        }

        double Optimizer::learn(const Matrix &x, const Matrix &ans) {
            auto lg = m_net->flow(x, ans);
            auto dv = get_update(lg.gradient);
            m_net->update(dv);
            return lg.loss;
        }

        Optimizer::HyperParam Optimizer::hyper_param() const {
            return m_hp;
        }

        Optimizer::Avg Optimizer::average() const {
            return m_avg;
        }

        Optimizer::ParamVector Optimizer::get_update(const Optimizer::ParamVector &grad) {
            return get_gradient_step(grad);
        }

        Optimizer::ParamVector Optimizer::get_gradient_step(const Optimizer::ParamVector &grad) {
            ParamVector ret;

            for(const auto &param: grad){
                Param here;
                for(const auto &pair: param)
                    here[pair.first] = pair.second*m_hp[HyperParamKey::lr]*-1.;
                ret.push_back(here);
            }

            return ret;
        }
    }
}