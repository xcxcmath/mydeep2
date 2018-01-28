#include "Optimizer.h"

namespace mydeep {
    namespace optimizer {

        Optimizer::Optimizer(network::Network *net, double learning_rate)
                :m_net(net)
        {
            m_hp[HyperParamKey::lr] = learning_rate;
        }

        Optimizer::Optimizer(network::Network *net,
                             const HyperParam &hp,
                             const Avg &avg)
                :m_net(net),
                 m_hp(hp),
                 m_avg(avg)
        {

        }

        double Optimizer::learn(const Matrix &x, const Matrix &ans) {
            m_in = x; m_ans = ans;
            auto lg = m_net->flow(x, ans);
            auto dv = get_update(lg.gradient);
            m_net->update(dv);
            return lg.loss;
        }

        HyperParam Optimizer::hyper_param() const {
            return m_hp;
        }

        Optimizer::Avg Optimizer::average() const {
            return m_avg;
        }

        ParamVector Optimizer::get_update(const ParamVector &grad) {
            return get_gradient_step(grad);
        }

        ParamVector Optimizer::get_gradient_step(const ParamVector &grad) {
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