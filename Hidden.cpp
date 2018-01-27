#include "Hidden.h"

namespace mydeep {
    namespace layer {
        Hidden::Hidden() = default;

        Hidden::Hidden(const Param &param)
            : m_param(param)
        {

        }

        Matrix Hidden::forward(const Matrix &x) {
            m_in = x;
            return m_out = predict(x);
        }

        void Hidden::update(const Param &param) {
            for(const auto &dv : param)
                m_param[dv.first] += dv.second;
        }

        Param Hidden::param() const {
            return m_param;
        }
    }
}