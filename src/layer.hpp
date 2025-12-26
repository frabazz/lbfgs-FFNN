#pragma once
#include "common.hpp"
#include "minimizer_base.hpp"
#include <autodiff/reverse/var.hpp>
#include <autodiff/reverse/var/eigen.hpp>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/src/Core/Map.h>
#include <memory>
#include <random>
#include <vector>

class DenseLayer {

  using WMatT = Eigen::Map<Eigen::Matrix<autodiff::var, Eigen::Dynamic, Eigen::Dynamic>>;
  using BVecT = Eigen::Map<Eigen::Matrix<autodiff::var, Eigen::Dynamic, 1>>;

private:
  const unsigned int in;
  const unsigned int out;

  std::unique_ptr<WMatT> W;
  std::unique_ptr<BVecT> b;
  bool isLast = false;

public:
  DenseLayer(unsigned int _in, unsigned int _out) : in(_in),
                                                    out(_out) {
    W = nullptr;
    b = nullptr;
  }

  DenseLayer(unsigned int _in, unsigned int _out, bool _isLast) : in(_in),
                                                                  out(_out) {
    W = nullptr;
    b = nullptr;
    isLast = _isLast;
  }

  void bindParams(autodiff::var *params) {
    W = std::make_unique<WMatT>(params, out, in);
    b = std::make_unique<BVecT>(params + in * out, out);
  };

  size_t getSize() {
    return (in * out + out);
  }

  autodiff::VectorXvar forward(autodiff::VectorXvar &input) {
    autodiff::VectorXvar x = (*W) * input + (*b);
    if (isLast)
      return x;
    return x.unaryExpr([](const autodiff::var &v) -> autodiff::var {
      return autodiff::reverse::detail::condition(v > 0.0, v, 0.0);
    });
  }
};

class Network {
private:
  size_t params_size;

public:
  size_t getSize() {
    return params_size;
  }

  Network() {
    layers = std::vector<DenseLayer>();
    params_size = 0;
    params = nullptr;
  }

  void addLayer(DenseLayer &&layer) {
    params_size += layer.getSize();
    layers.push_back(std::move(layer));
  }

  void bindParams() {
    params = std::make_unique<autodiff::VectorXvar>(params_size);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> dist(0.0, 0.5);

    for (int i = 0; i < params_size; ++i) {
      (*params)(i) = dist(gen);
    }

    size_t offset = 0;
    for (auto &layer : layers) {
      layer.bindParams(params->data() + offset);
      offset += layer.getSize();
    }
  }

  autodiff::VectorXvar forward(autodiff::VectorXvar &input) {
    autodiff::VectorXvar x = input;
    for (auto &layer : layers)
      x = layer.forward(x);
    return x;
  }

  void train(std::shared_ptr<MinimizerBase<Eigen::VectorXd, Eigen::MatrixXd>> minimizer_ptr) {
    autodiff::VectorXvar in1(8);
    in1 << 0.5, -0.2, 1.0, 0.1, -0.5, 0.8, 0.3, -0.1;
    autodiff::VectorXvar out1(4);
    out1 << 1, 2, 3, 4;

    VecFun<autodiff::VectorXvar, autodiff::var> f = [&, this](autodiff::VectorXvar v) -> autodiff::var {
      for (int i = 0; i < params_size; ++i)
        (*params)(i) = v(i);

      autodiff::var mse = 0;
      autodiff::VectorXvar y = this->forward(in1);
      for (int i = 0; i < out1.size(); ++i) {
        autodiff::var err = y(i) - out1(i);
        mse += err * err;
      }
      std::cout << "evaluated mse of: " << mse << std::endl;
      return mse;
    };

    Eigen::VectorXd params_d(params_size);
    for (int i = 0; i < params_size; ++i) {
      params_d(i) = val((*params)(i));
    }

    Eigen::VectorXd final_params = minimizer_ptr->solve(params_d, f);

    for (int i = 0; i < params_size; ++i) {
      (*params)(i) = final_params(i);
    }
    std::cout << "trained in " << minimizer_ptr->iterations() << " iters" << std::endl;
  }

private:
  std::vector<DenseLayer> layers;
  std::unique_ptr<autodiff::VectorXvar> params;
};
