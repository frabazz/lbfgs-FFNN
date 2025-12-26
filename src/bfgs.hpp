#pragma once

#include "common.hpp"
#include "minimizer_base.hpp"
#include <autodiff/reverse/var.hpp>
#include <autodiff/reverse/var/eigen.hpp>
#include <eigen3/Eigen/Eigen>

template <typename M>
constexpr bool isSparse = std::is_base_of_v<Eigen::SparseMatrixBase<M>, M>;

template <typename M>
using DefaultSolverT = typename std::conditional<
    isSparse<M>,
    Eigen::ConjugateGradient<M>,
    Eigen::LDLT<M>>::type;

/**
 * @brief BFGS (Broyden–Fletcher–Goldfarb–Shanno) minimizer.
 *
 * Implements a full-memory quasi-Newton method for unconstrained optimization.
 * The algorithm maintains and updates a dense approximation @p B of the Hessian
 * matrix, and uses it to compute search directions by solving
 * \f$ B p_k = -\nabla f(x_k) \f$.
 *
 * @tparam V Vector type (e.g. Eigen::VectorXd).
 * @tparam M Matrix type (e.g. Eigen::MatrixXd).
 * @tparam Solver if specified can be used to specify solver type  (e.g. Eigen::ConjugateGradient) and must must be passed to the constructor
 */
template <typename V, typename M, typename Solver = DefaultSolverT<M>>
class BFGS : public MinimizerBase<V, M> {
  using Base = MinimizerBase<V, M>;
  using Base::_B;
  using Base::_iters;
  using Base::_max_iters;
  using Base::_tol;

protected:
  static constexpr bool UseDefaultSolver = std::is_same_v<Solver, DefaultSolverT<M>>;

  /**
   * This shenanigan (SolverT) is used to provide
   * a default solver type, such as Eigen::ConjugateGradient<M>, if the Solver
   * template parameter is not explicitly specified by the user.
   *
   * @note **Custom Solver Requirement:** If a custom solver is specified, it must be
   * passed to the constructor **by reference** (e.g., const SolverType& solver)
   * to ensure the user fully controls the solver's initialization parameters.
   *
   * Passing by reference is necessary because Eigen
   * linear solvers (like GMRES) typically do not allow copy
   * construction or assignment.
   */
  using SolverT = typename std::conditional<
      UseDefaultSolver,
      Solver,
      Solver &>::type;

private:
  SolverT _solver;

public:
  BFGS()
  requires(UseDefaultSolver) {
    _solver = DefaultSolverT<M>();
  }

  BFGS(Solver &solver)
  requires(!UseDefaultSolver) : _solver(solver) {
  }

  /**
   * @brief Run the BFGS optimization method.
   *
   * Starting from an initial point @p x and an initial Hessian approximation
   * @p B, this method iteratively performs:
   *  - computation of a search direction by solving Bp = -∇f(x)
   *  - line search along @p p to determine a step length @p alpha
   *  - update of the iterate @p x using the standard BFGS formula.
   *
   * The loop stops when either:
   *  - the gradient norm falls below the tolerance, or
   *  - the maximum number of iterations is reached.
   *
   * @param x Initial guess for the minimizer (passed by value).
   * @param f Objective function to minimize, mapping V → double.
   * @param Gradient Function returning the gradient ∇f(x), mapping V → V.
   *
   * @return Final estimate of the minimizer.
   */
  V solve(V x, VecFun<V, double> &f, GradFun<V> &Gradient) override {

    for (_iters = 0; _iters < _max_iters && Gradient(x).norm() > _tol;
         ++_iters) {

      // Factorize B and check success
      _solver.compute(_B);
      check(_solver.info() == Eigen::Success, "conjugate gradient solver error");

      // Search direction: p = -B^{-1} ∇f(x)
      V p = _solver.solve(-Gradient(x));

      // Line search to determine step length alpha
      double alpha = 1.0;
      alpha = this->line_search(x, p, f, Gradient);

      // Step and new iterate
      V s = alpha * p; ///< s_k = x_{k+1} − x_k.
      V x_next = x + s;

      // Gradient difference
      V y = Gradient(x_next) - Gradient(x); ///< y_k = ∇f_{k+1} − ∇f_k.

      // BFGS update: B_{k+1} = B_k + (y yᵀ)/(yᵀ s) − (B s sᵀ B)/(sᵀ B s)
      M b_prod = _B * s;
      _B = _B + (y * y.transpose()) / (y.transpose() * s) -
           (b_prod * b_prod.transpose()) / (s.transpose() * _B * s);

      // Move to the next iterate
      x = x_next;
    }

    return x;
  }

  using MinimizerBase<V, M>::solve;
};
