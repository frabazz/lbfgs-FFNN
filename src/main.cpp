#include <iostream>
#include <autodiff/reverse/var.hpp>
#include <autodiff/reverse/var/eigen.hpp>

using namespace autodiff;

// Se vuoi usare VectorXd originale di Eigen
// #include <Eigen/Core> 

var f(const ArrayXvar& x) {
  return (x * x).sum(); // Nota: usa .sqrt() o autodiff::sqrt
}

int main() {
    VectorXvar x(5);
    x << 1, 2, 3, 4, 5;
    
    var y = f(x);
    Eigen::VectorXd dydx = gradient(y, x);
    
    std::cout << "dy/dx = \n" << dydx << std::endl;
    return 0;
}
