// Included after NodeBase and Nav aliases; Nav2 executes exactly one goal.
struct LookaheadManager : NodeBase {
  Handover state;
  rclcpp_action::Client<Nav>::SharedPtr nav;
  rclcpp::Publisher<Msg>::SharedPtr request, control;
  rclcpp::TimerBase::SharedPtr timer;
  bool asked = false, active = false, lookahead_sent = false,
       goal_accepted = false;
  double origin = 0, start_wall = 0, goal_started = 0, feedback_remaining = -1,
         feedback_distance = -1;
  double nominal_speed = 1;
  int segments = 0, goal_epoch = 0;
  M points;
  J candidate;
  std::map<std::pair<int, int>, double> durations;
  LookaheadManager()
      : NodeBase("manager"), points(matrix(read("candidates.json"))) {
    nominal_speed = read("protocol.json")["navigation_nominal_speed_m_s"];
    nav = rclcpp_action::create_client<Nav>(this, "/navigate_to_pose");
    request = publisher("/cl/request");
    control = publisher("/cl/control");
    subscribe("/cl/plan", [this](J j) {
      record(j);
      if (j["status"] == "ready") {
        for (auto &e : read("graph.json"))
          durations[{int(e["from"]), int(e["to"])}] = e["duration_s"];
        active = true;
        origin = now().seconds();
        start_wall = wall();
        send(control, J{{"cmd", "start"}, {"origin", origin}});
        ask_search(false, now().seconds() - origin);
      } else if (j["status"] == "error") {
        if (active)
          stop("planner_error");
      } else if (j["status"] == "plan" && active) {
        if (now().seconds() - origin >= WINDOW_S) {
          stop("window_completed");
          return;
        }
        if (!state.accept(j.value("request_id", -1), j.value("anchor", -1))) {
          record(J{{"event", "stale_plan_rejected"},
                   {"request_id", j.value("request_id", -1)}});
          return;
        }
        if (j.value("kind", std::string()) == "search") {
          candidate = j;
          record(J{{"event", "candidate_ready"},
                   {"while_navigating", state.moving},
                   {"anchor", j["anchor"]},
                   {"request_id", j["request_id"]},
                   {"round", j["round"]}});
          if (!state.moving)
            activate_candidate();
        } else {
          if (state.moving)
            throw std::runtime_error("Activation during active Nav2 goal");
          dispatch(j["route"].get<std::vector<int>>());
        }
      }
    });
    timer = create_wall_timer(std::chrono::milliseconds(50), [this] {
      if (!asked && now().seconds() > 1 &&
          request->get_subscription_count() > 0 &&
          nav->action_server_is_ready()) {
        asked = true;
        send(request, J{{"cmd", "prepare"}});
      }
      if (!active)
        return;
      if (now().seconds() - origin >= WINDOW_S) {
        stop("window_completed");
        return;
      }
      if (state.moving && goal_accepted && !lookahead_sent &&
          !state.outstanding && now().seconds() - goal_started >= 1) {
        double nominal = std::max(0., durations[{state.current, state.target}] -
                                          (now().seconds() - goal_started));
        double remaining =
            feedback_remaining > 0 && std::isfinite(feedback_remaining)
                ? feedback_remaining
                : nominal;
        remaining = std::max(remaining,
                             std::max(0., feedback_distance) / nominal_speed);
        double predicted =
            std::min(WINDOW_S, now().seconds() - origin + remaining);
        lookahead_sent = true;
        record(J{{"event", "lookahead_requested"},
                 {"from", state.target},
                 {"predicted_start_t", predicted},
                 {"nav2_remaining_s", feedback_remaining},
                 {"nominal_remaining_s", nominal},
                 {"goal_epoch", goal_epoch}});
        ask_search(true, predicted);
      }
    });
  }
  void ask_search(bool ahead, double start) {
    int anchor = ahead ? state.target : state.current,
        id = state.begin_request(anchor);
    send(request, J{{"cmd", "plan"},
                    {"current", anchor},
                    {"origin", origin},
                    {"request_id", id},
                    {"lookahead", ahead},
                    {"predicted_start_t", start},
                    {"goal_epoch", goal_epoch}});
  }
  void activate_candidate() {
    if (candidate.is_null() || state.moving || state.outstanding)
      return;
    if (int(candidate["anchor"]) != state.current) {
      record(J{{"event", "stale_candidate_rejected"}});
      candidate = J();
      return;
    }
    int id = state.begin_request(state.current);
    record(J{{"event", "handover_validation_requested"},
             {"anchor", state.current},
             {"candidate_gp_version", candidate["gp_version"]},
             {"candidate_round", candidate["round"]}});
    send(request, J{{"cmd", "activate"},
                    {"current", state.current},
                    {"origin", origin},
                    {"request_id", id},
                    {"route", candidate["route"]},
                    {"lookahead", false},
                    {"goal_epoch", goal_epoch}});
    candidate = J();
  }
  void dispatch(const std::vector<int> &route) {
    if (now().seconds() - origin >= WINDOW_S) {
      stop("window_completed");
      return;
    }
    if (route.empty() || route.front() != state.current)
      throw std::runtime_error("Dispatch anchor mismatch");
    int next = -1;
    for (int v : route)
      if (v != state.current) {
        next = v;
        break;
      }
    if (next < 0) {
      record(J{{"event", "no_motion_route"}});
      return;
    }
    state.depart(next);
    lookahead_sent = false;
    goal_accepted = false;
    goal_started = now().seconds();
    feedback_remaining = -1;
    feedback_distance = -1;
    int epoch = ++goal_epoch;
    Nav::Goal g;
    g.pose.header.frame_id = "map";
    g.pose.header.stamp = now();
    g.pose.pose.position.x = points(next, 0) - 48.5;
    g.pose.pose.position.y = points(next, 1) - 32;
    g.pose.pose.orientation.w = 1;
    rclcpp_action::Client<Nav>::SendGoalOptions opts;
    opts.goal_response_callback = [this, epoch](auto handle) {
      if (!active || epoch != goal_epoch)
        return;
      if (!handle) {
        record(J{{"event", "goal_rejected"}});
        stop("goal_rejected");
      } else
        goal_accepted = true;
    };
    opts.feedback_callback =
        [this, epoch](auto, const std::shared_ptr<const Nav::Feedback> f) {
          if (!active || epoch != goal_epoch)
            return;
          feedback_remaining = f->estimated_time_remaining.sec +
                               f->estimated_time_remaining.nanosec * 1e-9;
          feedback_distance = f->distance_remaining;
        };
    opts.result_callback = [this, next, epoch](auto r) {
      record(J{{"event", "navigation_result"},
               {"code", int(r.code)},
               {"target", next},
               {"goal_epoch", epoch}});
      if (!active || epoch != goal_epoch)
        return;
      if (r.code != rclcpp_action::ResultCode::SUCCEEDED) {
        stop("navigation_error");
        return;
      }
      state.arrive(next);
      segments++;
      record(J{{"event", "handover_arrived"},
               {"target", next},
               {"waiting_for_search", state.outstanding},
               {"goal_epoch", epoch}});
      if (!candidate.is_null())
        activate_candidate();
      else if (!state.outstanding)
        ask_search(false, now().seconds() - origin);
    };
    record(J{{"event", "goal"},
             {"target", next},
             {"x", points(next, 0)},
             {"y", points(next, 1)},
             {"goal_epoch", epoch}});
    nav->async_send_goal(g, opts);
  }
  void stop(const std::string &status) {
    active = false;
    nav->async_cancel_all_goals();
    send(control, J{{"cmd", "stop"}});
    save("mission.json", J{{"observation_sim_s", now().seconds() - origin},
                           {"wall_s", wall() - start_wall},
                           {"origin", origin},
                           {"configured_window_s", WINDOW_S},
                           {"return_home_required", false},
                           {"segments", segments},
                           {"status", status},
                           {"execution_mode", "lookahead"},
                           {"search_outstanding_at_end", state.outstanding}});
    record(J{{"event", "stopped"}, {"status", status}});
  }
};
