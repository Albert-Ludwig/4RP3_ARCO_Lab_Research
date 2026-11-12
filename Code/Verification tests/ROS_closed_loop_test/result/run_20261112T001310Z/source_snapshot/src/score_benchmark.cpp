// Controlled old-vs-batched comparison on identical saved GPs, events and
// weights.
#include "core.hpp"
#include "matrix_threads.hpp"
#include <iostream>
#include <type_traits>
namespace scalar_reference {
#include "../../result/run_20261111T212025Z/source_snapshot/src/core.hpp"
}
J read_json(const std::string &p) {
  std::ifstream f(p);
  J j;
  f >> j;
  return j;
}
template <typename G, typename E, typename S>
J measure(const std::string &root, bool coarse = false) {
  J results = J::array();
  auto graph = read_json(root + "/graph.json");
  for (int round = 1; round <= 3; ++round) {
    auto p = read_json(root + "/plan_" + std::to_string(round) + ".json");
    G gp(read_json(root + "/gp_" + std::to_string(round) + ".json"));
    std::vector<std::vector<E>> edges(31, std::vector<E>(31));
    for (int i = 0; i < 31; ++i) {
      edges[i][i].xy = M::Zero(2, 2);
      edges[i][i].length = V::Zero(2);
      edges[i][i].duration = 0;
    }
    for (auto &row : graph) {
      auto &e = edges[int(row["from"])][int(row["to"])];
      e.xy = matrix(row["xy"]);
      e.length = V::Zero(e.xy.rows());
      for (int k = 1; k < e.xy.rows(); ++k)
        e.length(k) = e.length(k - 1) + (e.xy.row(k) - e.xy.row(k - 1)).norm();
      // Case 1 retains the previous 0.25 m/s events; 2/3 use current 1 m/s.
      e.duration = double(row["duration_s"]) * (round == 1 ? 1. : .25);
    }
    double t = p["t"], begin = wall();
    auto search = [&]() {
      if constexpr (std::is_same_v<S, Search>)
        return S(gp, edges, t, 300 - t, coarse ? 2. : .5);
      else
        return S(gp, edges, t, 300 - t);
    }();
    double setup = wall() - begin;
    search.deadline = wall() + 600;
    auto route = p["search_route"].template get<std::vector<int>>();
    J scores = J::array(), times = J::array();
    auto evaluate = [&](const std::vector<int> &r) {
      double start = wall(), value = search.score(r);
      times.push_back(wall() - start);
      scores.push_back(value);
    };
    evaluate(route); // Cold event-feature cache.
    evaluate(route); // Route-score cache hit.
    auto prefix = route;
    prefix.pop_back();
    evaluate(prefix); // Shared event prefix.
    search.scores.clear();
    evaluate(route); // Warm feature cache, recomputed score.
    results.push_back(J{{"matrix_threads", configure_matrix_threads()},
                        {"case", round},
                        {"gp_observations", gp.x.rows()},
                        {"nominal_speed_m_s", round == 1 ? .25 : 1.},
                        {"elapsed_s", t},
                        {"route", route},
                        {"events", array(search.events(route))},
                        {"setup_s", setup},
                        {"scores", scores},
                        {"timings_s", times},
                        {"scoring_inputs", search.q.rows()}});
  }
  return results;
}
int main(int argc, char **argv) {
  configure_matrix_threads();
  if (argc != 4)
    return 2;
  auto data =
      std::string(argv[1]) == "scalar"
          ? measure<scalar_reference::GP, scalar_reference::Edge,
                    scalar_reference::Search>(argv[2])
          : measure<GP, Edge, Search>(argv[2], std::string(argv[1]) == "grid2");
  std::ofstream(argv[3]) << data.dump(2);
}
