#pragma once

#include "common.hpp"
#include <eigen3/Eigen/Cholesky>
#include <eigen3/Eigen/Eigen>
#include <eigen3/Eigen/IterativeLinearSolvers>
#include <autodiff/reverse/var.hpp>
#include <autodiff/reverse/var/eigen.hpp>

/**
 * @brief Base class for iterative minimization algorithms.
 *
 * @tparam V Type used for vectors (e.g. Eigen::VectorXd).
 * @tparam M Type used for matrices (e.g. Eigen::MatrixXd).
 * @tparam Solver type used for solver (e.g. Eigen::ConjugateGradient)
 */
template <typename V, typename M>
class MinimizerBase {
private:
public:
  /// Virtual destructor to allow proper cleanup in derived classes.
  virtual ~MinimizerBase() = default;

  /**
   * @brief Get the number of iterations performed by the last solve().
   *
   * @return Number of iterations actually performed.
   */
  int iterations() const noexcept {
    return _iters;
  }

  /**
   * @brief Get the current tolerance used as stopping criterion.
   *
   * @return Tolerance on the stopping condition.
   */
  double tolerance() const noexcept {
    return _tol;
  }

  /**
   * @brief Set the maximum number of iterations allowed in solve().
   *
   * @param max_iters Maximum number of iterations.
   */
  void setMaxIterations(int max_iters) noexcept {
    _max_iters = max_iters;
  }

  /**
   * @brief Set the tolerance used as stopping criterion.
   *
   * Typically used as threshold on gradient norm or relative error.
   *
   * @param tol New tolerance value.
   */
  void setTolerance(double tol) noexcept { _tol = tol; }

  /**
   * @brief Set the initial guess for the Hessian matrix
   *
   * @param b Initial Hessian guess.
   */
  void setInitialHessian(M b) noexcept { _B = b; }

  /**
   * @brief Set the Hessian function
   *
   * @param hessFun Function object returning the Hessian matrix.
   */
  void setHessian(const HessFun<V, M> &hessFun) noexcept { _hessFun = hessFun; }
  /**
   * @brief Solve the minimization problem given an initial guess.
   *
   * This is the main entry point of any concrete minimization algorithm
   * inheriting from this base class.
   *
   * @param x Initial guess for the minimizer; can be used as in/out.
   * @param f Objective function to minimize, mapping V -> double.
   * @param Gradient Function object returning the gradient of f, mapping V -> V.
   *
   * @return Approximate minimizer of the function f.
   */
  virtual V solve(V x, VecFun<V, double> &f, GradFun<V> &Gradient) = 0;
  

protected:
  /// Maximum number of iterations allowed in the optimization loop.
  unsigned int _max_iters = 1000;

  /// Number of iterations performed in the last call to solve().
  unsigned int _iters = 0;

  /// Tolerance used as stopping criterion.
  double _tol = 1.e-10;

  /// Hessian guess
  M _B;

  HessFun<V, M> _hessFun;

  /// Maximum number of iterations allowed in Armijo line search (if used).
  double armijo_max_iter = 20;

  /// Maximum number of iterations allowed in generic line search.
  double max_line_iters = 50;

  /// Memory size parameter (e.g. for L-BFGS methods).
  size_t m = 15;

  /// Initial step size guess for Wolfe line search.
  double alpha_wolfe = 1e-3;

  /// Parameter c1 in the Wolfe/Armijo condition.
  double c1 = 1e-4;

  /// Parameter c2 in the Wolfe curvature condition.
  double c2 = 0.9;

  /// Contraction factor used when shrinking the step size.
  double rho = 0.5;

  /**
   * @brief Perform a line search to find a suitable step length alpha.
   *
   * This routine attempts to find a step length @p alpha along direction @p p
   * starting from point @p x such that (approximate) Wolfe conditions are
   * satisfied:
   * - sufficient decrease condition (controlled by @ref c1)
   * - curvature condition (controlled by @ref c2)
   *
   * @param x Current point.
   * @param p Search direction.
   * @param f Objective function to minimize.
   * @param Gradient Function object returning the gradient of f.
   *
   * @return Step length alpha found by the line search. If no suitable alpha
   *         is found within @ref max_line_iters, the last tested alpha is
   *         returned as a fallback.
   */
  double line_search(V x, V p, VecFun<V, double> &f, GradFun<V> &Gradient) {
    double f_old = f(x);
    double grad_f_old = Gradient(x).dot(p);

    double inf = std::numeric_limits<double>::infinity();
    double alpha_min = 0.0;
    double alpha_max = inf;

    double alpha = 1.0;

    for (int i = 0; i < max_line_iters; ++i) {
      V x_new = x + alpha * p;
      double f_new = f(x_new);

      // Armijo (sufficient decrease) condition
      if (f_new > f_old + c1 * alpha * grad_f_old) {
        alpha_max = alpha;
        alpha = rho * (alpha_min + alpha_max);
        continue;
      }

      double grad_f_new_dot_p = Gradient(x_new).dot(p);

      // Curvature condition
      if (grad_f_new_dot_p < c2 * grad_f_old) {
        alpha_min = alpha;
        if (alpha_max == inf)
          alpha *= 2;
        else
          alpha = rho * (alpha_min + alpha_max);

        continue;
      }
      return alpha;
    }
    // Fallback: If no alpha is found, return the last one
    return alpha;
  }
};
