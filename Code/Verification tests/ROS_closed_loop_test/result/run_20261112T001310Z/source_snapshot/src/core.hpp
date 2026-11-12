#pragma once
// Fixed 3D ARD RBF in metres, metres, seconds. All variances are degC squared.
// One planning worker owns mutable GP state. Search reads frozen GP/weights.
#include <Eigen/Dense>
#include <chrono>
#include <fstream>
#include <functional>
#include <map>
#include <nlohmann/json.hpp>
#include <vector>
inline constexpr double WINDOW_S = 300.;
using J = nlohmann::json;
using M = Eigen::MatrixXd;
using V = Eigen::VectorXd;
inline double wall() {
  return std::chrono::duration<double>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}
inline M matrix(const J &a) {
  M m(a.size(), a.empty() ? 0 : a[0].size());
  for (int i = 0; i < m.rows(); i++)
    for (int j = 0; j < m.cols(); j++)
      m(i, j) = a[i][j];
  return m;
}
inline J array(const M &m) {
  J a = J::array();
  for (int i = 0; i < m.rows(); i++) {
    J r = J::array();
    for (int j = 0; j < m.cols(); j++)
      r.push_back(m(i, j));
    a.push_back(r);
  }
  return a;
}
struct GP {
  M x, L;
  V y, alpha;
  Eigen::Vector3d length;
  double variance, noise, mean;
  int version = 0;
  GP(const J &j) {
    x = matrix(j["x"]);
    y = V::Zero(j["y"].size());
    for (int i = 0; i < y.size(); i++)
      y(i) = j["y"][i];
    for (int k = 0; k < 3; k++)
      length(k) = j["length"][k];
    variance = j["variance"];
    noise = j["noise"];
    mean = j["mean"];
    factor();
  }
  M kernel(const M &a, const M &b) const {
    M k(a.rows(), b.rows());
    for (int i = 0; i < a.rows(); i++)
      for (int j = 0; j < b.rows(); j++)
        k(i, j) =
            variance *
            std::exp(-.5 *
                     ((a.row(i) - b.row(j)).transpose().cwiseQuotient(length))
                         .squaredNorm());
    return k;
  }
  void factor() {
    M k = kernel(x, x);
    k.diagonal().array() += noise;
    Eigen::LLT<M> f(k);
    if (f.info() != Eigen::Success)
      throw std::runtime_error("GP Cholesky failed");
    L = f.matrixL();
    alpha = L.transpose().triangularView<Eigen::Upper>().solve(
        L.triangularView<Eigen::Lower>().solve((y.array() - mean).matrix()));
  }
  M features(const M &q) const {
    return L.triangularView<Eigen::Lower>().solve(kernel(x, q));
  }
  std::pair<V, V> posterior(const M &q) const {
    M k = kernel(x, q), v = L.triangularView<Eigen::Lower>().solve(k);
    return {V::Constant(q.rows(), mean) + k.transpose() * alpha,
            (V::Constant(q.rows(), variance) -
             v.array().square().colwise().sum().transpose().matrix())
                .cwiseMax(0)};
  }
  void append(const J &rows) {
    if (rows.empty())
      return;
    int n = x.rows();
    x.conservativeResize(n + rows.size(), 3);
    y.conservativeResize(n + rows.size());
    for (size_t i = 0; i < rows.size(); i++) {
      x.row(n + i) << double(rows[i]["x"]), double(rows[i]["y"]),
          double(rows[i]["t"]);
      y(n + i) = rows[i]["temperature"];
    }
    factor();
    version++;
  }
  J snapshot() const {
    std::vector<double> values(y.data(), y.data() + y.size());
    return J{{"x", array(x)},
             {"y", values},
             {"length", {length(0), length(1), length(2)}},
             {"variance", variance},
             {"noise", noise},
             {"mean", mean},
             {"version", version}};
  }
};
struct Edge {
  M xy;
  V length;
  double duration = INFINITY;
};
struct Search {
  GP &gp;
  std::vector<std::vector<Edge>> &edges;
  M q, qfeatures;
  V weights;
  double started, deadline, elapsed, cap, eval_step;
  int calls = 0, hits = 0;
  int feature_batches = 0, feature_misses = 0;
  double feature_seconds = 0, objective_seconds = 0;
  bool timed_out = false;
  std::map<std::vector<int>, double> scores;
  struct Feature {
    V gp, cross;
  };
  std::map<std::array<double, 3>, Feature> cache;
  std::vector<int> best;
  double best_gain = 0;
  Search(GP &g, std::vector<std::vector<Edge>> &e, double t, double horizon,
         double evaluation_step = .5)
      : gp(g), edges(e), elapsed(t), cap(horizon), eval_step(evaluation_step) {
    started = wall();
    deadline = started + 45;
    if (!(std::isfinite(eval_step) && eval_step > 0 && eval_step <= WINDOW_S) ||
        std::abs(std::round(WINDOW_S / eval_step) * eval_step - WINDOW_S) >
            1e-8)
      throw std::runtime_error(
          "Scoring time step must divide the mission window");
    // 70 exact-area cells, including clipped right/top boundary cells.
    M spatial(70, 3);
    V areas(70);
    int i = 0;
    for (int x = 0; x < 97; x += 10)
      for (int y = 0; y < 64; y += 10) {
        double w = std::min(10, 97 - x), h = std::min(10, 64 - y);
        spatial.row(i) << x + w / 2, y + h / 2, t;
        areas(i++) = w * h;
      }
    V u = gp.posterior(spatial).second;
    V w = areas.cwiseProduct(u);
    if (u.maxCoeff() <= 1e-12)
      w = areas;
    const int nt = int(std::round(WINDOW_S / eval_step)) + 1;
    q.resize(70 * nt, 3);
    weights.resize(q.rows());
    for (int k = 0; k < nt; k++)
      for (int j = 0; j < 70; j++) {
        q.row(k * 70 + j) << spatial(j, 0), spatial(j, 1), k * eval_step;
        weights(k * 70 + j) = w(j);
      }
    qfeatures = gp.features(q);
    weights /= weights.sum();
  }
  double finish(const std::vector<int> &r) const {
    double t = elapsed;
    for (size_t i = 1; i < r.size(); i++)
      if (r[i] >= 0)
        t += edges[r[i - 1]][r[i]].duration;
    return t;
  }
  M events(const std::vector<int> &r) const {
    std::vector<Eigen::Vector3d> out;
    double left = elapsed;
    int tick = int(std::floor(elapsed + 1e-9)) + 1;
    for (size_t i = 1; i < r.size(); i++) {
      if (r[i] < 0)
        break; // Zero-cost virtual terminal: finish at any real checkpoint.
      const auto &e = edges[r[i - 1]][r[i]];
      double end = left + e.duration;
      if (!std::isfinite(end))
        throw std::runtime_error("Unreachable route");
      while (tick <= std::min(end, WINDOW_S - 1e-9) + 1e-10) {
        if (e.duration > 0) {
          double dist =
              (tick - left) / e.duration * e.length(e.length.size() - 1);
          int k = 1;
          while (k < e.length.size() - 1 && e.length(k) < dist)
            k++;
          double f = (dist - e.length(k - 1)) /
                     std::max(1e-12, e.length(k) - e.length(k - 1));
          Eigen::Vector2d p = (1 - f) * e.xy.row(k - 1).transpose() +
                              f * e.xy.row(k).transpose();
          out.push_back({p(0), p(1), double(tick)});
        }
        tick++;
      }
      left = end;
    }
    M a(out.size(), 3);
    for (size_t i = 0; i < out.size(); i++)
      a.row(i) = out[i];
    return a;
  }
  double score(const std::vector<int> &r) {
    if (wall() > deadline) {
      timed_out = true;
      throw std::runtime_error("search_timeout");
    }
    auto it = scores.find(r);
    if (it != scores.end()) {
      hits++;
      return it->second;
    }
    M a = events(r);
    if (!a.rows())
      return scores[r] = 0;
    calls++;
    M z(gp.x.rows(), a.rows()), cross(q.rows(), a.rows());
    double feature_start = wall();
    // Copy cached columns first: later eviction must not invalidate a hit.
    std::vector<int> missing;
    for (int i = 0; i < a.rows(); ++i) {
      std::array<double, 3> key{a(i, 0), a(i, 1), a(i, 2)};
      auto cached = cache.find(key);
      if (cached == cache.end()) {
        missing.push_back(i);
      } else {
        z.col(i) = cached->second.gp;
        cross.col(i) = cached->second.cross;
      }
    }
    // Same conditional covariance, evaluated with multiple right-hand sides:
    // Z = L^-1 K(X,A); C = K(Q,A) - F(Q)^T Z.
    // Limit temporary memory to 32 event columns rather than an entire route.
    constexpr int batch_columns = 32;
    for (size_t offset = 0; offset < missing.size(); offset += batch_columns) {
      int count = std::min<size_t>(batch_columns, missing.size() - offset);
      M points(count, 3);
      for (int k = 0; k < count; ++k)
        points.row(k) = a.row(missing[offset + k]);
      M features = gp.features(points);
      M conditional = gp.kernel(q, points);
      conditional.noalias() -= qfeatures.transpose() * features;
      ++feature_batches;
      feature_misses += count;
      for (int k = 0; k < count; ++k) {
        int column = missing[offset + k];
        z.col(column) = features.col(k);
        cross.col(column) = conditional.col(k);
        std::array<double, 3> key{a(column, 0), a(column, 1), a(column, 2)};
        if (cache.size() >= 4096)
          cache.erase(cache.begin());
        cache.emplace(key, Feature{features.col(k), conditional.col(k)});
      }
    }
    feature_seconds += wall() - feature_start;
    double objective_start = wall();
    M covariance = gp.kernel(a, a) - z.transpose() * z;
    covariance = (covariance + covariance.transpose()).eval() * .5;
    covariance.diagonal().array() += gp.noise;
    Eigen::LLT<M> chol(covariance);
    if (chol.info() != Eigen::Success)
      throw std::runtime_error("Event Cholesky failed");
    M solved = chol.matrixL().solve(cross.transpose());
    V reduction = solved.array().square().colwise().sum().transpose();
    double value = weights.dot(reduction);
    if (!std::isfinite(value) || value < -1e-8)
      throw std::runtime_error("Invalid score");
    objective_seconds += wall() - objective_start;
    double accepted = std::max(0., value);
    scores[r] = accepted;
    // Every fully scored prefix is a feasible open route. Preserve it before
    // deeper recursion can hit the wall guard (not only after a root branch).
    if (accepted > best_gain && finish(r) <= elapsed + cap + 1e-8) {
      best = r;
      best_gain = accepted;
    }
    return accepted;
  }
  std::vector<int> recurse(std::vector<int> prefix, int end, double end_s,
                           int depth) {
    int start = prefix.back();
    auto base = prefix;
    if (!(prefix.size() == 1 && start == end))
      base.push_back(end);
    if (finish(base) > end_s + 1e-8)
      return {};
    if (!depth)
      return base;
    auto choice = base;
    double gain = score(base);
    for (int v = 0; v < int(edges.size()); v++) {
      auto p = prefix;
      p.push_back(v);
      if (depth == 1) {
        p.push_back(end);
        if (finish(p) <= end_s + 1e-8) {
          double val = score(p);
          if (val > gain + 1e-12) {
            gain = val;
            choice = p;
          }
        }
      } else {
        double lo = finish(p),
               hi = end_s - (end < 0 ? 0. : edges[v][end].duration);
        if (lo > hi + 1e-8)
          continue;
        for (int split = 1; split <= 2; split++) {
          auto left = recurse(prefix, v, lo + (hi - lo) * split / 3, depth - 1);
          if (left.empty())
            continue;
          auto right = recurse(left, end, end_s, depth - 1);
          if (right.empty())
            continue;
          double val = score(right);
          if (val > gain + 1e-12) {
            gain = val;
            choice = right;
            if (depth == 2 && val > best_gain) {
              best = right;
              best_gain = val;
            }
          }
        }
      }
    }
    return choice;
  }
  J run(int current) {
    best = {current};
    // A virtual terminal (-1) is reachable at zero cost from any checkpoint.
    // It is never a navigation goal or a sensing event.
    try {
      auto completed = recurse({current}, -1, elapsed + cap, 2);
      if (!completed.empty()) {
        double value = score(completed);
        if (value >= best_gain) {
          best = completed;
          best_gain = value;
        }
      }
    } catch (const std::exception &e) {
      if (std::string(e.what()) != "search_timeout")
        throw;
    }
    if (!best.empty() && best.back() == -1)
      best.pop_back();
    return J{{"route", best},
             {"score", best_gain},
             {"timeout", timed_out},
             {"calls", calls},
             {"cache_hits", hits},
             {"feature_batches", feature_batches},
             {"feature_misses", feature_misses},
             {"feature_seconds", feature_seconds},
             {"objective_seconds", objective_seconds},
             {"feature_batch_columns", 32},
             {"scoring_time_step_s", eval_step},
             {"scoring_input_count", q.rows()},
             {"seconds", wall() - started},
             {"events", array(events(best))},
             {"weights", std::vector<double>(weights.data(),
                                             weights.data() + weights.size())}};
  }
};
