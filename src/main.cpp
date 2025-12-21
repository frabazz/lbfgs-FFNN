#include <iostream>
#include <autodiff/reverse/var.hpp>
#include <autodiff/reverse/var/eigen.hpp>
#include "bfgs.hpp"

using namespace autodiff;

using Vec = Eigen::VectorXd;
using Mat = Eigen::MatrixXd;

int main() {    

    VecFun<VectorXvar, var> f = [](VectorXvar v) {
      var val = 0.0;
      double A = 10.0;
      int n = v.size();
      for (int i = 0; i < n; ++i) {
        val += (v(i) * v(i)) - (A * cos(2.0 * M_PI * v(i)));
      }
      return A * n + val;
    };

    VecFun<VectorXvar, var> f_rosenbrock = [](VectorXvar v) {
    var val = 0.0;
    int n = v.size();
    for (int i = 0; i < n - 1; ++i) {
        var term1 = 100.0 * pow((v(i+1) - v(i) * v(i)), 2);
        var term2 = pow((1.0 - v(i)), 2);
        val += term1 + term2;
    }
    return val;
};

    
    auto solver = BFGS<Vec, Mat>();
    int n = 4;
    Vec x(n);
    for(int i = 0;i < n;++i)
      x(i) = -2;

    Mat m(n,n);
    for(int i = 0;i < n; ++i)
      m(i,i) = 1;
    
    solver.setMaxIterations(4000);
    solver.setTolerance(1.e-10);
    solver.setInitialHessian(m);

    
    Vec sol = solver.solve(x, f_rosenbrock);
    std::cout << "=====SOL====" << std::endl;
    std::cout << sol << std::endl;
    std::cout << "iters: " << solver.iterations() << std::endl;
    return 0;
}
