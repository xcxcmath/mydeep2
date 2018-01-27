#ifndef MYDEEP2_LAYER_H
#define MYDEEP2_LAYER_H

#include "Core.h"

namespace mydeep {
    class Layer {
    public:
        virtual ~Layer();
        virtual Matrix predict(const Matrix &x) = 0;

    protected:
        Matrix m_in;
        Matrix m_out;
    };
}

#endif //MYDEEP2_LAYER_H
