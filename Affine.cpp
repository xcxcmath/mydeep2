#include "Affine.h"

namespace mydeep {
    namespace layer {
        Affine::Affine(Index out_size, Affine::InitKey init)
                : m_init(init)
        {
            m_param[ParamKey::weight] = m_param[ParamKey::bias]
                                      = Matrix::Zero(out_size, 1);
            m_param[ParamKey::init_stddev] = Matrix::Ones(1, 1);
            m_param[ParamKey::initialized] = Matrix::Zero(1, 1);
        }

        Affine::Affine(Index out_size, double stddev)
                : m_init(InitKey::custom)
        {
            assert(stddev >= 0 && "Standard deviation must be over zero.");

            m_param[ParamKey::weight] = m_param[ParamKey::bias]
                                      = Matrix::Zero(out_size, 1);
            m_param[ParamKey::init_stddev] = Matrix::Constant(1, 1, stddev);
            m_param[ParamKey::initialized] = Matrix::Zero(1, 1);
        }

        Affine::Affine(const Param &param)
                : Hidden(param),
                  m_init(InitKey::none)
        {

        }

        Matrix Affine::predict(const Matrix &x) {
            if(m_param[ParamKey::initialized](0, 0) == 0)
                initialize(x.rows());

            return (m_param[ParamKey::weight] * x).colwise() +
                   m_param[ParamKey::bias].col(0);
        }

        BackOutput Affine::backward(const Matrix &delta) {
            m_backout.delta = m_param[ParamKey::weight].transpose() * delta;
            m_backout.gradient[ParamKey::weight] = delta * m_in.transpose();
            m_backout.gradient[ParamKey::bias] = delta.rowwise().sum();

            return m_backout;
        }

        void Affine::initialize(Index in_size) {
            auto &w = m_param[ParamKey::weight];

            const auto out_size = w.rows();

            w = Matrix::Zero(out_size, in_size);

            auto &v = m_param[ParamKey::init_stddev] = Matrix::Ones(1, 1);
            switch(m_init){
                case InitKey::he:
                    v = Matrix::Constant(1, 1, std::sqrt(2. / in_size));
                    break;
                case InitKey::xavier:
                    v = Matrix::Constant(1, 1, std::sqrt(1. / in_size));
                    break;
                default:
                    break;
            }

            std::random_device rd;
            std::normal_distribution<double> dist(0., v(0, 0));
            w = w.unaryExpr([&rd, &dist](const double&){return dist(rd);});

            m_param[ParamKey::initialized](0, 0) = 1.;
        }

    }
}