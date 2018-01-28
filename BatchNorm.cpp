#include "BatchNorm.h"

namespace mydeep {
    namespace layer {
        BatchNorm::BatchNorm(double momentum, double eps) {
            assert(eps > 0 && "eps must be over zero.");

            auto &p = m_param[ParamKey::momentum_epsilon] = Matrix::Zero(1, 2);
            p(0, 0) = momentum;
            p(0, 1) = eps;
            m_param[ParamKey::initialized] = Matrix::Zero(1, 1);
        }

        BatchNorm::BatchNorm(const Param &param)
                : Hidden(param)
        {

        }

        Matrix BatchNorm::predict(const Matrix &x) {
            const double INIT = m_param[ParamKey::initialized](0, 0);
            assert(INIT > 0 && "BatchNorm layer is not prepared.");

            const auto &p = m_param[ParamKey::momentum_epsilon];
            const auto eps = p(0, 1);

            const auto &g = m_param[ParamKey::gamma];
            const auto &b = m_param[ParamKey::beta];
            const auto &cache_mean = m_param[ParamKey::mean];
            const auto &cache_var = m_param[ParamKey::var];

            Matrix norm = x.colwise() - cache_mean.col(0);
            const auto var_sqr = (cache_var.array()+eps).pow(-0.5).matrix().col(0);
            norm = var_sqr.asDiagonal() * norm;
            norm = (g.col(0).asDiagonal() * norm).colwise() + b.col(0);
            //TODO : asDiagonal() vs array()

            return norm;
        }

        Matrix BatchNorm::forward(const Matrix &x) {
            m_in = x;

            const double INIT = m_param[ParamKey::initialized](0, 0);
            if(INIT == 0){
                m_param[ParamKey::beta] = Matrix::Zero(x.rows(),1);
                m_param[ParamKey::gamma] = Matrix::Ones(x.rows(), 1);
                m_param[ParamKey::mean] = Matrix::Zero(x.rows(), 1);
                m_param[ParamKey::var] = Matrix::Zero(x.rows(), 1);
                m_norm = Matrix::Zero(x.rows(), x.cols());

                m_param[ParamKey::initialized](0, 0) = 1.;
            }

            auto &cache_mean = m_param[ParamKey::mean];
            auto &cache_var = m_param[ParamKey::var];
            const auto &g = m_param[ParamKey::gamma];
            const auto &b = m_param[ParamKey::beta];
            const auto &momentum = m_param[ParamKey::momentum_epsilon](0, 0);
            const auto &eps = m_param[ParamKey::momentum_epsilon](0, 1);

            m_mean = x.rowwise().mean();
            m_xc = x.colwise() - m_mean.col(0);
            m_var = m_xc.rowwise().squaredNorm()/x.cols();
            const auto var_sqr = (m_var.array() + eps).pow(-0.5).matrix().col(0);

            cache_mean = cache_mean * momentum + m_mean * (1-momentum);
            cache_var = cache_var * momentum + m_var * (1-momentum);

            m_norm = var_sqr.asDiagonal() * m_xc;
            m_out = (g.col(0).asDiagonal() * m_norm).colwise() + b.col(0);

            return m_out;
        }

        BackOutput BatchNorm::backward(const Matrix &delta) {
            m_backout.gradient[ParamKey::beta] = delta.rowwise().sum();
            m_backout.gradient[ParamKey::gamma] = delta.cwiseProduct(m_norm).rowwise().sum();

            const auto cols = delta.cols();
            const auto var_eps = m_var.array() + m_param[ParamKey::momentum_epsilon](0, 1);

            const auto var_1_5 = var_eps.pow(-1.5).matrix().col(0);
            const auto var_sqr = var_eps.pow(-0.5).matrix().col(0);

            const auto dnorm = m_param[ParamKey::gamma].col(0).asDiagonal() * delta;
            const Matrix dvar = (var_1_5.asDiagonal() * dnorm.cwiseProduct(m_xc) * -0.5).rowwise().sum();
            Matrix dmean = (var_sqr.asDiagonal() * dnorm * -1.).rowwise().sum();
            dmean = dmean.colwise() +
                    (dvar.col(0).asDiagonal() * (m_xc.rowwise().mean() * -2.)).col(0);

            m_backout.delta = (var_sqr.asDiagonal() * dnorm).colwise() + dmean.col(0) / cols;
            m_backout.delta = m_backout.delta + (dvar.col(0).asDiagonal() * m_xc * 2. /cols);

            return m_backout;
        }
    }
}