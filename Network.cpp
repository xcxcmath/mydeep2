#include "Network.h"

namespace mydeep {
    namespace network {

        Network::Network()
                : m_out(nullptr)
        {

        }

        Network::~Network() {
            for(auto &layer : m_layers)
                delete layer;
            delete m_out;
        }

        Matrix Network::predict(const Matrix &x) {
            auto temp = x;
            for(auto &layer : m_layers)
                temp = layer->predict(temp);
            return m_out->predict(temp);
        }

        Network::LossGradient Network::flow(const Matrix &x, const Matrix &ans) {
            auto temp = x;

            for(auto &layer : m_layers)
                temp = layer->forward(temp);

            const double loss = m_out->forward(temp, ans);

            temp = m_out->backward(ans);

            ParamVector gradient;

            for(auto it = m_layers.rbegin(); it != m_layers.rend(); ++it){
                const auto here = (*it)->backward(temp);
                temp = here.delta;
                gradient.push_back(here.gradient);
            }

            std::reverse(gradient.begin(), gradient.end());

            return {loss, gradient};
        }

        void Network::update(const Network::ParamVector &pv) {
            for(size_t i = 0; i < pv.size(); ++i)
                m_layers[i]->update(pv[i]);
        }

        void Network::insert(layer::Hidden *layer) {
            m_layers.push_back(layer);
        }

        void Network::insert(layer::Output *layer) {
            delete m_out;
            m_out = layer;
        }
    }
}