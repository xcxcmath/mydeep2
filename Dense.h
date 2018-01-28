#ifndef MYDEEP2_DENSE_H
#define MYDEEP2_DENSE_H

#include "Affine.h"
#include "Activation.h"
#include "Dropout.h"
#include "BatchNorm.h"
#include "NAG.h"
#include "Adam.h"

#define ACTIVATION new mydeep::layer::Activation
#define RELU new mydeep::layer::Activation(mydeep::layer::ReLU)
#define SIGMOID new mydeep::layer::Activation(mydeep::layer::Sigmoid)
#define TANH new mydeep::layer::Activation(mydeep::layer::Tanh)
#define AFFINE new mydeep::layer::Affine
#define DROPOUT new mydeep::layer::Dropout
#define BATCHNORM new mydeep::layer::BatchNorm

#define OUTPUT new mydeep::layer::Output
#define SOFTMAX new mydeep::layer::Output(mydeep::layer::Softmax)
#define IDENTITY new mydeep::layer::Output(mydeep::layer::Identity)

#endif //MYDEEP2_DENSE_H
