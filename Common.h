#ifndef MYDEEP2_CORE_H
#define MYDEEP2_CORE_H

#include <vector>
#include <map>
#include <memory>
#include <functional>
#include <random>
#include <algorithm>
#include <cassert>
#include <Core>

namespace mydeep {
    using Matrix = Eigen::MatrixXd;
    using Index = Eigen::Index;
    using UnaryFunction = std::function<double(const double &)>;

    UnaryFunction cwise_pow(double power);
    UnaryFunction cwise_exp();
    UnaryFunction cwise_log();
}

#endif //MYDEEP2_CORE_H
