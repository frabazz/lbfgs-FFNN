#pragma once

#include "common.hpp"
#include "minimizer_base.hpp"
#include <autodiff/forward/dual.hpp>
#include <eigen3/Eigen/Eigen>

/**
 * @brief Limited-memory BFGS (L-BFGS) minimizer.
 *
 * Implements a quasi-Newton optimization method with limited memory, storing
 * only the most recent curvature pairs (s_k, y_k) to approximate the inverse
 * Hessian. Suitable for large-scale unconstrained optimization problems.
 *
 * @tparam V Vector type (e.g., Eigen::VectorXd).
 * @tparam M Matrix type (e.g., Eigen::MatrixXd).
 */
template <typename V, typename M>
class LBFGS : public MinimizerBase<V, M> {
  using Base = MinimizerBase<V, M>;
  using Base::_iters;
  using Base::_max_iters;
  using Base::_tol;
  using Base::alpha_wolfe;
  using Base::m;

public:
  /**
   * @brief Perform the L-BFGS optimization on the objective function f.
   *
   * Starting from an initial guess @p x, this method iteratively computes
   * search directions using L-BFGS two-loop recursion and performs a
   * line search that satisfies Wolfe conditions.
   *
   * @param x Initial guess for the minimizer (passed by value).
   * @param f Objective function to minimize, mapping V → double.
   * @param Gradient Function returning the gradient ∇f(x), mapping V → V.
   *
   * @return The final estimate of the minimizer.
   */
  V solve(V x, VecFun<V, double> &f, GradFun<V> &Gradient) override {

    std::vector<V> s_list;        ///< Stored displacement vectors s_k = x_{k+1} − x_k.
    std::vector<V> y_list;        ///< Stored gradient differences y_k = ∇f_{k+1} − ∇f_k.
    std::vector<double> rho_list; ///< Scalars ρ_k = 1 / (y_kᵀ s_k).

    V grad = Gradient(x); ///< Current gradient.
    V p = -grad;          ///< Initial descent direction.
    V x_new = x;          ///< Updated point.

    for (_iters = 0; _iters < _max_iters; ++_iters) {

      // Stopping condition based on gradient norm
      if (grad.norm() < _tol) {
        break;
      }

      // Compute L-BFGS search direction
      p = compute_direction(grad, s_list, y_list, rho_list);

      // Wolfe line search to select step length
      alpha_wolfe = this->line_search(x, p, f, Gradient);

      // Point update
      x_new = x + alpha_wolfe * p;
      V s = x_new - x;

      // Gradient update
      V grad_new = Gradient(x_new);
      V y = grad_new - grad;

      x = x_new;

      // Store curvature pair (s_k, y_k)
      double rho = 1.0 / y.dot(s);
      s_list.push_back(s);
      y_list.push_back(y);
      rho_list.push_back(rho);

      // Enforce memory limit m
      if (s_list.size() > m) {
        s_list.erase(s_list.begin());
        y_list.erase(y_list.begin());
        rho_list.erase(rho_list.begin());
      }

      grad = grad_new;
    }

    return x;
  }

  /**
   * @brief Compute the L-BFGS search direction using the two-loop recursion.
   *
   * Approximates the action of the inverse Hessian on the gradient using
   * the stored curvature pairs (s_k, y_k). This avoids forming or storing
   * the full Hessian matrix.
   *
   * @param grad Current gradient vector.
   * @param s_list Stored displacement vectors s_k.
   * @param y_list Stored gradient differences y_k.
   * @param rho_list Stored scaling terms ρ_k = 1 / (y_kᵀ s_k).
   *
   * @return Search direction p_k, typically a descent direction.
   */
  V compute_direction(const V &grad,
                      const std::vector<V> &s_list,
                      const std::vector<V> &y_list,
                      const std::vector<double> &rho_list) {

    // If no curvature information is available, fall back to steepest descent
    if (s_list.empty()) {
      return -grad;
    }

    V z = V::Zero(grad.size());
    V q = grad;
    std::vector<double> alpha_list(s_list.size());

    // First loop: backward pass
    for (int i = static_cast<int>(s_list.size()) - 1; i >= 0; --i) {
      alpha_list[i] = rho_list[i] * s_list[i].dot(q);
      q -= alpha_list[i] * y_list[i];
    }

    // Scaling of the initial Hessian approximation H0 = γ I
    double gamma = s_list.back().dot(y_list.back()) /
                   y_list.back().dot(y_list.back());

    // Apply H0
    z = gamma * q;

    // Second loop: forward pass
    for (size_t i = 0; i < s_list.size(); ++i) {
      double beta = rho_list[i] * y_list[i].dot(z);
      z += s_list[i] * (alpha_list[i] - beta);
    }

    // Final search direction
    return -z;
  }

  using MinimizerBase<V, M>::solve;

private:
};
