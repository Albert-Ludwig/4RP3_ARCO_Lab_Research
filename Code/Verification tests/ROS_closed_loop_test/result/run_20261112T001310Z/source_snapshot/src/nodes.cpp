#include "core.hpp"
#include "handover.hpp"
#include "matrix_threads.hpp"
#include <atomic>
#include <cstdio>
#include <future>
#include <mutex>
#include <nav2_msgs/action/compute_path_to_pose.hpp>
#include <nav2_msgs/action/navigate_to_pose.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_action/rclcpp_action.hpp>
#include <std_msgs/msg/string.hpp>
#include <thread>
using Msg = std_msgs::msg::String;
using PathAction = nav2_msgs::action::ComputePathToPose;
using Nav = nav2_msgs::action::NavigateToPose;
std::string out = std::getenv("CL_OUTPUT") ? std::getenv("CL_OUTPUT") : ".";
J read(const std::string &f) {
  std::ifstream s(out + "/" + f);
  J j;
  s >> j;
  return j;
}
void save(const std::string &f, const J &j) {
  std::ofstream(out + "/" + f) << j.dump(2);
}
struct NodeBase : rclcpp::Node {
  std::ofstream log;
  std::mutex log_mutex;
  std::vector<rclcpp::Subscription<Msg>::SharedPtr> subs;
  NodeBase(std::string name) : Node(name), log(out + "/" + name + ".jsonl") {
    set_parameter(rclcpp::Parameter("use_sim_time", true));
  }
  auto publisher(std::string topic) {
    return create_publisher<Msg>(topic, 100);
  }
  void send(rclcpp::Publisher<Msg>::SharedPtr p, const J &j) {
    Msg m;
    m.data = j.dump();
    p->publish(m);
  }
  void subscribe(std::string t, std::function<void(J)> cb) {
    subs.push_back(
        create_subscription<Msg>(t, 100, [this, cb](Msg::SharedPtr m) {
          try {
            cb(J::parse(m->data));
          } catch (std::exception &e) {
            record(J{{"error", e.what()}});
          }
        }));
  }
  void record(J j) {
    std::lock_guard<std::mutex> l(log_mutex);
    j["wall"] = wall();
    j["sim"] = now().seconds();
    log << j.dump() << "\n";
    log.flush();
  }
};
struct Sampler : NodeBase {
  double origin = -1, x = 48.5, y = 32, pose_t = 0;
  int tick = 0;
  bool active = false;
  rclcpp::Publisher<Msg>::SharedPtr query, measurement;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr pose;
  rclcpp::TimerBase::SharedPtr timer;
  Sampler() : NodeBase("sampler") {
    query = publisher("/cl/query");
    measurement = publisher("/cl/measurement");
    pose = create_subscription<nav_msgs::msg::Odometry>(
        "/simulation/ground_truth/odom", rclcpp::SensorDataQoS(),
        [this](nav_msgs::msg::Odometry::SharedPtr m) {
          x = m->pose.pose.position.x + 48.5;
          y = m->pose.pose.position.y + 32;
          pose_t = rclcpp::Time(m->header.stamp).seconds();
        });
    subscribe("/cl/control", [this](J j) {
      if (j["cmd"] == "start") {
        origin = j["origin"];
        active = true;
      }
      if (j["cmd"] == "stop")
        active = false;
    });
    subscribe("/cl/temperature", [this](J j) {
      j["roundtrip_ms"] = (wall() - double(j["request_wall"])) * 1000;
      record(j);
      if (j.value("ok", false))
        send(measurement, j);
    });
    timer = create_wall_timer(std::chrono::milliseconds(10), [this] {
      if (!active)
        return;
      double t = now().seconds() - origin;
      int due = int(std::floor(t + 1e-9));
      if (due > tick && t < WINDOW_S) {
        int missed = due - tick - 1;
        tick = due;
        J j{{"id", due},
            {"t", t},
            {"x", x},
            {"y", y},
            {"request_wall", wall()},
            {"missed_ticks", missed},
            {"pose_age_s", now().seconds() - pose_t}};
        record(J{{"event", "query"}, {"record", j}});
        if (j["pose_age_s"].get<double>() < .5)
          send(query, j);
        else
          record(J{{"event", "stale_pose"}, {"id", due}});
      }
    });
  }
};
struct Planner : NodeBase {
  GP gp;
  M points;
  std::vector<std::vector<Edge>> edges;
  J buffer = J::array();
  std::mutex mutex;
  rclcpp::Publisher<Msg>::SharedPtr reply;
  rclcpp_action::Client<PathAction>::SharedPtr path;
  std::thread worker;
  std::atomic<bool> busy{false};
  int rounds = 0;
  double gp_available_since_t = 0;
  Planner()
      : NodeBase("planner"), gp(read("initial.json")),
        points(matrix(read("candidates.json"))) {
    reply = publisher("/cl/plan");
    path =
        rclcpp_action::create_client<PathAction>(this, "/compute_path_to_pose");
    subscribe("/cl/measurement", [this](J j) {
      std::lock_guard<std::mutex> l(mutex);
      buffer.push_back(j);
    });
    subscribe("/cl/request", [this](J j) {
      if (busy) {
        record(J{{"event", "busy_request_rejected"}});
        send(reply, J{{"status", "error"},
                      {"error", "Unexpected concurrent planning request"}});
        return;
      }
      if (worker.joinable())
        worker.join();
      busy = true;
      std::remove((out + "/planner_idle.json").c_str());
      worker = std::thread([this, j] {
        try {
          if (j["cmd"] == "prepare")
            prepare();
          else
            plan(j);
        } catch (std::exception &e) {
          record(J{{"event", "error"}, {"error", e.what()}});
          save("planner_idle.json",
               J{{"idle", true}, {"sim", now().seconds()}});
          busy = false;
          send(reply, J{{"status", "error"}, {"error", e.what()}});
        }
      });
    });
  }
  ~Planner() {
    if (worker.joinable())
      worker.join();
  }
  geometry_msgs::msg::PoseStamped pose(int i) {
    geometry_msgs::msg::PoseStamped p;
    p.header.frame_id = "map";
    p.header.stamp = now();
    p.pose.position.x = points(i, 0) - 48.5;
    p.pose.position.y = points(i, 1) - 32;
    p.pose.orientation.w = 1;
    return p;
  }
  void prepare() {
    if (!path->wait_for_action_server(std::chrono::seconds(60)))
      throw std::runtime_error("Nav2 planner unavailable");
    edges.resize(points.rows(), std::vector<Edge>(points.rows()));
    J graph = J::array();
    int count = 0;
    double begin = wall();
    double nominal_speed =
        read("protocol.json")["navigation_nominal_speed_m_s"];
    if (!(nominal_speed > 0))
      throw std::runtime_error("Invalid nominal navigation speed");
    for (int i = 0; i < points.rows(); i++)
      for (int j = 0; j < points.rows(); j++) {
        auto &e = edges[i][j];
        if (i == j) {
          e.xy.resize(2, 2);
          e.xy.row(0) = e.xy.row(1) = points.row(i);
          e.length = V::Zero(2);
          e.duration = 0;
          continue;
        }
        PathAction::Goal goal;
        goal.start = pose(i);
        goal.goal = pose(j);
        goal.use_start = true;
        goal.planner_id = "GridBased";
        auto f = path->async_send_goal(goal);
        if (f.wait_for(std::chrono::seconds(10)) != std::future_status::ready)
          throw std::runtime_error("Path goal timeout");
        auto handle = f.get();
        if (!handle)
          throw std::runtime_error("Path goal rejected");
        auto result = path->async_get_result(handle);
        if (result.wait_for(std::chrono::seconds(15)) !=
            std::future_status::ready)
          throw std::runtime_error("Path result timeout");
        auto r = result.get();
        if (r.code != rclcpp_action::ResultCode::SUCCEEDED ||
            r.result->path.poses.size() < 2)
          throw std::runtime_error("Candidate path unreachable");
        auto ps = r.result->path.poses;
        e.xy.resize(ps.size(), 2);
        e.length = V::Zero(ps.size());
        for (size_t k = 0; k < ps.size(); k++) {
          e.xy.row(k) << ps[k].pose.position.x + 48.5,
              ps[k].pose.position.y + 32;
          if (k)
            e.length(k) =
                e.length(k - 1) + (e.xy.row(k) - e.xy.row(k - 1)).norm();
        }
        e.duration = e.length(e.length.size() - 1) / nominal_speed;
        graph.push_back(J{{"from", i},
                          {"to", j},
                          {"xy", array(e.xy)},
                          {"duration_s", e.duration}});
        if (++count % 100 == 0)
          record(J{{"event", "paths"}, {"count", count}});
      }
    save("graph.json", graph);
    save("gp_0.json", gp.snapshot());
    record(J{
        {"event", "prepared"}, {"seconds", wall() - begin}, {"paths", count}});
    save("planner_idle.json", J{{"idle", true}, {"sim", now().seconds()}});
    busy = false;
    send(reply, J{{"status", "ready"}});
  }
  void plan(J req) {
    double start = wall();
    double started_t = now().seconds() - double(req["origin"]);
    const bool lookahead = req.value("lookahead", false);
    const bool activate = req.value("cmd", std::string()) == "activate";
    J batch = J::array();
    {
      std::lock_guard<std::mutex> l(mutex);
      if (!lookahead) {
        batch = buffer;
        buffer = J::array();
      }
    }
    GP updated = gp;
    try {
      updated.append(batch);
    } catch (...) {
      std::lock_guard<std::mutex> lock(mutex);
      for (auto &row : batch)
        buffer.push_back(row);
      throw;
    }
    gp = std::move(updated);
    double update_s = wall() - start;
    if (!batch.empty())
      gp_available_since_t = now().seconds() - double(req["origin"]);
    int round = ++rounds;
    save("batch_" + std::to_string(round) + ".json", batch);
    save("gp_" + std::to_string(round) + ".json", gp.snapshot());
    double t = now().seconds() - double(req["origin"]);
    const double predicted_start = req.value("predicted_start_t", t);
    t = std::max(t, predicted_start);
    double horizon = std::max(0., WINDOW_S - t);
    double scoring_dt = read("protocol.json").value("scoring_time_step_s", .5);
    Search search(gp, edges, t, horizon, scoring_dt);
    J ans;
    if (activate) {
      auto route = req["route"].get<std::vector<int>>();
      if (route.empty() || route.front() != int(req["current"]))
        throw std::runtime_error("Activation anchor mismatch");
      while (route.size() > 1 && search.finish(route) > WINDOW_S)
        route.pop_back();
      double value = search.score(route);
      ans = J{{"route", route},
              {"score", value},
              {"timeout", false},
              {"calls", search.calls},
              {"cache_hits", search.hits},
              {"feature_batches", search.feature_batches},
              {"feature_misses", search.feature_misses},
              {"feature_seconds", search.feature_seconds},
              {"objective_seconds", search.objective_seconds},
              {"feature_batch_columns", 32},
              {"scoring_time_step_s", scoring_dt},
              {"scoring_input_count", search.q.rows()},
              {"seconds", wall() - search.started},
              {"events", array(search.events(route))},
              {"weights", std::vector<double>(search.weights.data(),
                                              search.weights.data() +
                                                  search.weights.size())}};
    } else {
      ans = search.run(req["current"]);
    }
    // Refresh the selected route's event timestamps after the expensive search.
    // Frozen weights still belong to the same committed GP round.
    ans["search_score"] = ans["score"];
    ans["search_seconds"] = ans["seconds"];
    ans["search_events"] = ans["events"];
    search.elapsed =
        std::max(now().seconds() - double(req["origin"]), predicted_start);
    search.scores.clear();
    search.deadline = wall() + 10;
    auto selected = ans["route"].get<std::vector<int>>();
    ans["search_route"] = selected;
    while (selected.size() > 1 && search.finish(selected) > WINDOW_S)
      selected.pop_back();
    ans["route"] = selected;
    ans["trimmed_after_search"] =
        selected != ans["search_route"].get<std::vector<int>>();
    ans["execution_event_start_t"] = search.elapsed;
    ans["predicted_route_finish_t"] = search.finish(selected);
    ans["score"] = search.score(selected);
    ans["events"] = array(search.events(selected));
    ans["selected_route_rescore_s"] =
        wall() - start - update_s - double(ans["search_seconds"]);
    ans["status"] = "plan";
    ans["kind"] = activate ? "activate" : "search";
    ans["request_id"] = req.value("request_id", 0);
    ans["anchor"] = req["current"];
    ans["lookahead"] = lookahead;
    ans["goal_epoch"] = req.value("goal_epoch", 0);
    ans["planning_started_t"] = started_t;
    ans["gp_committed_t"] = gp_available_since_t;
    ans["requested_predicted_start_t"] = predicted_start;
    ans["round"] = round;
    ans["gp_version"] = gp.version;
    ans["update_s"] = update_s;
    ans["batch_size"] = batch.size();
    ans["t"] = t;
    ans["finish_t"] = now().seconds() - double(req["origin"]);
    ans["total_s"] = wall() - start;
    save("plan_" + std::to_string(round) + ".json", ans);
    record(ans);
    ans.erase("weights");
    ans.erase("events");
    save("planner_idle.json", J{{"idle", true}, {"sim", now().seconds()}});
    busy = false;
    send(reply, ans);
  }
};
struct SerialManager : NodeBase {
  rclcpp_action::Client<Nav>::SharedPtr nav;
  rclcpp::Publisher<Msg>::SharedPtr request, control;
  rclcpp::TimerBase::SharedPtr timer;
  bool prepared = false, asked = false, active = false, ended = false;
  double origin = 0, start_wall = 0;
  int current = 0, segments = 0;
  M points;
  SerialManager()
      : NodeBase("manager"), points(matrix(read("candidates.json"))) {
    nav = rclcpp_action::create_client<Nav>(this, "/navigate_to_pose");
    request = publisher("/cl/request");
    control = publisher("/cl/control");
    subscribe("/cl/plan", [this](J j) {
      record(j);
      if (j["status"] == "ready") {
        prepared = true;
        active = true;
        origin = now().seconds();
        start_wall = wall();
        send(control, J{{"cmd", "start"}, {"origin", origin}});
        ask();
      } else if (j["status"] == "plan" && active) {
        auto route = j["route"].get<std::vector<int>>();
        int next = -1;
        for (int v : route)
          if (v != current) {
            next = v;
            break;
          }
        if (next < 0) {
          record(J{{"event", "no_motion_route"}});
          return;
        }
        Nav::Goal g;
        g.pose.header.frame_id = "map";
        g.pose.header.stamp = now();
        g.pose.pose.position.x = points(next, 0) - 48.5;
        g.pose.pose.position.y = points(next, 1) - 32;
        g.pose.pose.orientation.w = 1;
        rclcpp_action::Client<Nav>::SendGoalOptions opts;
        opts.result_callback = [this, next](auto r) {
          record(J{{"event", "navigation_result"},
                   {"code", int(r.code)},
                   {"target", next}});
          if (active && r.code == rclcpp_action::ResultCode::SUCCEEDED) {
            current = next;
            segments++;
            ask();
          }
        };
        record(J{{"event", "goal"},
                 {"target", next},
                 {"x", points(next, 0)},
                 {"y", points(next, 1)}});
        nav->async_send_goal(g, opts);
      }
    });
    timer = create_wall_timer(std::chrono::milliseconds(50), [this] {
      if (!asked && now().seconds() > 1 &&
          request->get_subscription_count() > 0 &&
          nav->action_server_is_ready()) {
        asked = true;
        send(request, J{{"cmd", "prepare"}});
      }
      if (active && now().seconds() - origin >= WINDOW_S) {
        active = false;
        ended = true;
        nav->async_cancel_all_goals();
        send(control, J{{"cmd", "stop"}});
        save("mission.json", J{{"observation_sim_s", now().seconds() - origin},
                               {"wall_s", wall() - start_wall},
                               {"origin", origin},
                               {"configured_window_s", WINDOW_S},
                               {"return_home_required", false},
                               {"segments", segments},
                               {"status", "window_completed"}});
        record(J{{"event", "stopped"}});
      }
    });
  }
  void ask() {
    send(request, J{{"cmd", "plan"}, {"origin", origin}, {"current", current}});
  }
};
#include "lookahead_manager.hpp"
int main(int argc, char **argv) {
  auto threads = configure_matrix_threads();
  if (argc > 1 &&
      (std::string(argv[1]) == "audit" || std::string(argv[1]) == "planner"))
    save("matrix_threads.json", threads);
  if (argc > 1 && std::string(argv[1]) == "audit") {
    GP gp(read("initial.json"));
    M q(3, 3);
    q << 10, 10, 0, 48.5, 32, 60, 80, 50, 120;
    auto p = gp.posterior(q);
    save("cpp_audit.json",
         J{{"queries", array(q)},
           {"mean", std::vector<double>(p.first.data(),
                                        p.first.data() + p.first.size())},
           {"variance",
            std::vector<double>(p.second.data(),
                                p.second.data() + p.second.size())}});
    // Regression: free-endpoint GRG must work when no return edge exists.
    J tiny = read("initial.json");
    tiny["x"] = J::array({J::array({0., 0., -1.})});
    tiny["y"] = J::array({double(tiny["mean"])});
    GP fixture(tiny);
    std::vector<std::vector<Edge>> e(3, std::vector<Edge>(3));
    for (int i = 0; i < 3; ++i) {
      e[i][i].xy = M::Zero(2, 2);
      e[i][i].length = V::Zero(2);
      e[i][i].duration = 0;
    }
    auto link = [&](int i, int j, double x0, double y0, double x1, double y1) {
      auto &a = e[i][j];
      a.xy.resize(2, 2);
      a.xy << x0, y0, x1, y1;
      a.length.resize(2);
      a.length << 0, std::hypot(x1 - x0, y1 - y0);
      a.duration = a.length(1);
    };
    link(0, 1, 0, 0, 4, 0);
    link(1, 2, 4, 0, 4, 3);
    link(0, 2, 0, 0, 4, 3);
    Search test(fixture, e, 0, 8);
    auto ans = test.run(0);
    auto route = ans["route"].get<std::vector<int>>();
    if (route.size() < 2 || route.back() == 0 ||
        std::find(route.begin(), route.end(), -1) != route.end() ||
        test.finish(route) > 8)
      throw std::runtime_error("Free-endpoint regression failed");
    save("open_route_check.json", J{{"passed", true},
                                    {"route", route},
                                    {"return_edges_available", false},
                                    {"duration_s", test.finish(route)}});
    return 0;
  }
  rclcpp::init(argc, argv);
  std::shared_ptr<rclcpp::Node> n;
  std::string role = argc > 1 ? argv[1] : "manager";
  if (role == "sampler")
    n = std::make_shared<Sampler>();
  else if (role == "planner")
    n = std::make_shared<Planner>();
  else if (read("protocol.json")
               .value("execution_mode", std::string("serial")) == "lookahead")
    n = std::make_shared<LookaheadManager>();
  else
    n = std::make_shared<SerialManager>();
  rclcpp::spin(n);
  rclcpp::shutdown();
  return 0;
}
