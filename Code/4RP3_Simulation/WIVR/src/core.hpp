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
    if (x.rows() == 0)
      x.resize(0, 3);
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
    if (x.rows() == 0)
      return;
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
    if (x.rows() == 0)
      return M(0, q.rows());
    return L.triangularView<Eigen::Lower>().solve(kernel(x, q));
  }
  std::pair<V, V> posterior(const M &q) const {
    if (x.rows() == 0)
      return {V::Constant(q.rows(), mean), V::Constant(q.rows(), variance)};
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
    J values = J::array();
    for (int i = 0; i < y.size(); ++i)
      values.push_back(y(i));
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
  std::string policy;
  static constexpr int home = 0;
  // Returning to Home consumes mission time. Its observations are acquired during execution.
  static constexpr int terminal = -1;
  std::vector<std::vector<Edge>> &edges;
  M q, qfeatures;
  V weights;
  double started, deadline, elapsed, mission_horizon, cap, eval_step;
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
  Search(GP &g, std::vector<std::vector<Edge>> &e, const J &cfg, const std::string &method, double start_t=0, double weight_t=0)
      : gp(g), policy(method), edges(e), elapsed(start_t),
        mission_horizon(cfg["navigation_budget_s"]), cap(std::max(0., mission_horizon - start_t)),
        eval_step(cfg["scoring_time_step_s"]) {
    started = wall(); deadline = started + double(cfg["search_guard_s"]);
    if (policy != "mi" && policy != "emse" && policy != "wivr") throw std::runtime_error("Unknown policy");
    double width=cfg["extent_m"][0], height=cfg["extent_m"][1], cell=cfg["scoring_cell_m"];
    std::vector<Eigen::Vector3d> cells; std::vector<double> area;
    for(double x=0;x<width;x+=cell) for(double y=0;y<height;y+=cell) {
      double w=std::min(cell,width-x), h=std::min(cell,height-y);
      cells.push_back({x+w/2,y+h/2,weight_t});area.push_back(w*h);
    }
    M spatial(cells.size(),3); V areas(cells.size());
    for(int i=0;i<spatial.rows();i++){spatial.row(i)=cells[i].transpose();areas(i)=area[i];}
    V w=areas;
    if(policy=="wivr") {V u=gp.posterior(spatial).second;if(u.maxCoeff()>1e-12)w=areas.cwiseProduct(u);}
    int nt=int(std::round(mission_horizon/eval_step))+1;
    q.resize(spatial.rows()*nt,3);weights.resize(q.rows());
    for(int k=0;k<nt;k++)for(int j=0;j<spatial.rows();j++){
      int i=k*spatial.rows()+j;q.row(i)<<spatial(j,0),spatial(j,1),k*eval_step;weights(i)=w(j);
    }
    weights/=weights.sum();
    if(policy!="mi") qfeatures=gp.features(q);
  }
  double finish(const std::vector<int> &r) const {
    double t = elapsed;
    for (size_t i = 1; i < r.size(); i++)
      t += edges[r[i - 1]][r[i] == terminal ? home : r[i]].duration;
    return t;
  }
  M events(const std::vector<int> &r) const {
    std::vector<Eigen::Vector3d> out;
    double left = elapsed;
    int tick = int(std::floor(elapsed + 1e-9)) + 1;
    for (size_t i = 1; i < r.size(); i++) {
      if (r[i] < 0)
        break; // Return samples are outside the GRG objective.
      const auto &e = edges[r[i - 1]][r[i]];
      double end = left + e.duration;
      if (!std::isfinite(end))
        throw std::runtime_error("Unreachable route");
      while (tick <= std::min(end, mission_horizon - 1e-9) + 1e-10) {
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
    const bool conditioned = gp.x.rows() != 0;
    const bool uses_grid = policy != "mi";
    M z(gp.x.rows(), a.rows()), cross(uses_grid ? q.rows() : 0, a.rows());
    double feature_start = wall();
    // Copy cached columns first: later eviction must not invalidate a hit.
    std::vector<int> missing;
    for (int i = 0; i < a.rows(); ++i) {
      std::array<double, 3> key{a(i, 0), a(i, 1), a(i, 2)};
      auto cached = cache.find(key);
      if (cached == cache.end()) {
        missing.push_back(i);
      } else {
        if (conditioned)
          z.col(i) = cached->second.gp;
        if (uses_grid)
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
      M conditional(0,count);
      if (uses_grid) {
        conditional = gp.kernel(q, points);
        if (conditioned)
          conditional.noalias() -= qfeatures.transpose() * features;
      }
      ++feature_batches;
      feature_misses += count;
      for (int k = 0; k < count; ++k) {
        int column = missing[offset + k];
        Feature feature;
        if (conditioned) {
          feature.gp = features.col(k);
          z.col(column) = feature.gp;
        }
        if (uses_grid) {
          feature.cross = conditional.col(k);
          cross.col(column) = feature.cross;
        }
        std::array<double, 3> key{a(column, 0), a(column, 1), a(column, 2)};
        if (cache.size() >= 4096)
          cache.erase(cache.begin());
        cache.emplace(key, std::move(feature));
      }
    }
    feature_seconds += wall() - feature_start;
    double objective_start = wall();
    M covariance = gp.kernel(a, a);
    if (conditioned)
      covariance.noalias() -= z.transpose() * z;
    covariance = (covariance + covariance.transpose()).eval() * .5;
    covariance.diagonal().array() += gp.noise;
    Eigen::LLT<M> chol(covariance);
    if (chol.info() != Eigen::Success)
      throw std::runtime_error("Event Cholesky failed");
    double value;
    if(policy=="mi") {
      value=chol.matrixL().toDenseMatrix().diagonal().array().log().sum()-.5*a.rows()*std::log(gp.noise);
    } else {
      M solved=chol.matrixL().solve(cross.transpose());
      V reduction=solved.array().square().colwise().sum().transpose();value=weights.dot(reduction);
    }
    if (!std::isfinite(value) || value < -1e-8)
      throw std::runtime_error("Invalid score");
    objective_seconds += wall() - objective_start;
    double accepted = std::max(0., value);
    scores[r] = accepted;
    // Preserve complete routes to the virtual sampling terminal.
    if ((best.size()<2 || accepted>best_gain) && r.size()>2 && r.back()==terminal && finish(r)<=elapsed+cap+1e-8) {
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
    for (int v = home + 1; v < int(edges.size()); v++) {
      if(v==start || v==end) continue;
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
               hi = end_s - edges[v][end == terminal ? home : end].duration;
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
            if (end == terminal && val > best_gain) {
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
    // Plan the remaining sampling legs. The mission executor handles final return.
    try {
      auto completed = recurse({current}, terminal, elapsed + cap, 2);
      if (!completed.empty()) {
        double value = score(completed);
        if (completed.size()>2 && value>=best_gain) {
          best = completed;
          best_gain = value;
        }
      }
    } catch (const std::exception &e) {
      if (std::string(e.what()) != "search_timeout")
        throw;
    }
    // A late-time search may have no nontrivial budget-feasible route. Return it explicitly.
    if (best.back()==terminal) best.pop_back();
    return J{{"route", best},
             {"score", best_gain},
             {"policy", policy},
             {"predicted_navigation_s", finish(best)},
             {"predicted_home_arrival_s", finish(best) + edges[best.back()][home].duration},
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
