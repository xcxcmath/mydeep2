#include "Common.h"

namespace mydeep {
    UnaryFunction cwise_pow(double power) {
        return [power](const double &i){return std::pow(i, power);};
    }

    UnaryFunction cwise_exp() {
        return [](const double &i){return std::exp(i);};
    }

    UnaryFunction cwise_log() {
        return [](const double &i){return std::log(i);};
    }
}