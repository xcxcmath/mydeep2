#ifndef MYDEEP2_AFFINE_H
#define MYDEEP2_AFFINE_H

#include "Hidden.h"

namespace mydeep {
    namespace layer {
        class Affine : public Hidden {
        public:
            enum class InitKey {
                none, he, xavier, custom,
            };

            explicit Affine(Index out_size, InitKey init = InitKey::he);
            explicit Affine(Index out_size, double stddev);
            explicit Affine(const Param &param);

            Matrix predict(const Matrix &x) override;
            BackOutput backward(const Matrix &delta) override;

        protected:
            void initialize(Index in_size);

            const InitKey m_init;
        };
    }
}

#endif //MYDEEP2_AFFINE_H
