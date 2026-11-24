#include "core.hpp"
#include "matrix_threads.hpp"
#include <atomic>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <thread>
#include <nav2_msgs/action/compute_path_to_pose.hpp>
#include <nav2_msgs/action/navigate_to_pose.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_action/rclcpp_action.hpp>
#include <std_msgs/msg/string.hpp>
using Msg=std_msgs::msg::String;
using PathAction=nav2_msgs::action::ComputePathToPose;
using Nav=nav2_msgs::action::NavigateToPose;
std::string out=std::getenv("CL_OUTPUT");
J read(const std::string& f){J j;std::ifstream(out+"/"+f)>>j;return j;}
void save(const std::string& f,const J& j){std::ofstream(out+"/"+f)<<j.dump(2);}
struct Job {J state,meta;};
struct Experiment:rclcpp::Node {
  GP gp;J cfg,initial_plan,measurement_buffer=J::array();M points;std::vector<std::vector<Edge>> edges;
  std::ofstream log;std::mutex log_lock,queue_lock,result_lock;
  std::condition_variable condition;std::deque<Job>queue;std::map<int,J>results;
  std::thread preparation,calculation;std::atomic<int>ready{0},completed{0},requested{0};
  std::atomic<bool>stop_worker{false};size_t peak_queue=0; int coalesced=0; std::vector<int> retained_route; int retained_plan_id=0;
  rclcpp_action::Client<PathAction>::SharedPtr paths;
  rclcpp_action::Client<Nav>::SharedPtr nav;
  rclcpp_action::ClientGoalHandle<Nav>::SharedPtr goal_handle;
  rclcpp::Publisher<Msg>::SharedPtr query;
  rclcpp::Subscription<Msg>::SharedPtr temperature;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr pose;
  rclcpp::TimerBase::SharedPtr timer;
  bool preparing=false,active=false,finished=false,nav_active=false,waiting=false,returning=false,home_done=false;
  double x=15,y=10,pose_time=0,origin=0,started_wall=0,arrived_t=0,waiting_s=0;
  bool cancel_requested=false;
  double sampling_end=0,sampling_wall=0,return_start=-1,return_wall=0;
  double home_arrival_t=-1, home_error_at_arrival=0;
  int return_measurements=0;
  int sampling_stop_node=0;
  int sampling_segments=0,completed_at_sampling_end=0,started_at_sampling_end=0;
  bool stop_had_feasible_leg=false;
  std::string sampling_stop_reason;
  int current=0,target=0,epoch=0,tick=0,pending=0,received=0,missed=0,stale=0,segments=0,nav_failures=0;
  Experiment():Node("gp_grg_mi_latest_gp_test"),gp(read("initial.json")),cfg(read("protocol.json")),points(matrix(read("candidates.json"))),log(out+"/events.jsonl"){
    set_parameter(rclcpp::Parameter("use_sim_time",true));
    query=create_publisher<Msg>("/cl/query",100);
    paths=rclcpp_action::create_client<PathAction>(this,"/compute_path_to_pose");
    nav=rclcpp_action::create_client<Nav>(this,"/navigate_to_pose");
    pose=create_subscription<nav_msgs::msg::Odometry>("/simulation/ground_truth/odom",rclcpp::SensorDataQoS(),[this](nav_msgs::msg::Odometry::SharedPtr m){
      x=m->pose.pose.position.x+15;y=m->pose.pose.position.y+10;pose_time=rclcpp::Time(m->header.stamp).seconds();
    });
    temperature=create_subscription<Msg>("/cl/temperature",100,[this](Msg::SharedPtr msg){
      J row=J::parse(msg->data);pending--;row["roundtrip_ms"]=(wall()-double(row["request_wall"]))*1000;record("temperature",row);
      if(!row.value("ok",false)||!active)return;
      received++;
      if(row.value("returning",false))return_measurements++;
      measurement_buffer.push_back(row);
      int interval=cfg["update_interval_s"];
      if(int(row["id"])%interval==0) update_gp(false);

    });
    timer=create_wall_timer(std::chrono::milliseconds(10),[this]{try{poll();}catch(const std::exception&e){record("error",{{"message",e.what()}});finish("error");}});
  }
  ~Experiment(){stop_worker=true;condition.notify_all();if(preparation.joinable())preparation.join();if(calculation.joinable())calculation.join();}
  void record(const std::string&e,J j=J::object()){
    std::lock_guard<std::mutex>l(log_lock);j["event"]=e;j["wall"]=wall();j["sim"]=now().seconds();j["event_t"]=origin?now().seconds()-origin:0;log<<j.dump()<<"\n";log.flush();
  }
  geometry_msgs::msg::PoseStamped point(int i,int from=0){
    geometry_msgs::msg::PoseStamped p;p.header.frame_id="map";p.header.stamp=now();p.pose.position.x=points(i,0)-15;p.pose.position.y=points(i,1)-10;
    double a=std::atan2(points(i,1)-points(from,1),points(i,0)-points(from,0));p.pose.orientation.z=std::sin(a/2);p.pose.orientation.w=std::cos(a/2);return p;
  }
  double travel_time(double length_m) const {
    return length_m / double(cfg["travel_estimator_speed_m_s"]) +
           double(cfg["travel_estimator_segment_allowance_s"]);
  }
  void prepare(){try{
    J graph=J::array();int n=points.rows();edges.resize(n,std::vector<Edge>(n));
    std::ifstream cached(out+"/graph.json");
    if(cached.good()){
      cached>>graph;
      for(int i=0;i<n;i++){edges[i][i].xy.resize(2,2);edges[i][i].xy.row(0)=edges[i][i].xy.row(1)=points.row(i);edges[i][i].length=V::Zero(2);edges[i][i].duration=0;}
      for(auto&j:graph){
        auto&e=edges[int(j["from"])][int(j["to"])];
        e.xy=matrix(j["xy"]);e.length=V::Zero(e.xy.rows());
        for(int k=1;k<e.xy.rows();k++)e.length(k)=e.length(k-1)+(e.xy.row(k)-e.xy.row(k-1)).norm();
        e.duration=travel_time(e.length(e.length.size()-1));
        j["duration_s"]=e.duration;
      }
    } else
    for(int i=0;i<n;i++)for(int j=0;j<n;j++){
      auto&e=edges[i][j];
      if(i==j){e.xy.resize(2,2);e.xy.row(0)=e.xy.row(1)=points.row(i);e.length=V::Zero(2);e.duration=0;continue;}
      PathAction::Goal goal;goal.start=point(i);goal.goal=point(j,i);goal.use_start=true;goal.planner_id="GridBased";
      auto f=paths->async_send_goal(goal);if(f.wait_for(std::chrono::seconds(10))!=std::future_status::ready)throw std::runtime_error("Path goal timeout");auto h=f.get();if(!h)throw std::runtime_error("Path rejected");
      auto rf=paths->async_get_result(h);if(rf.wait_for(std::chrono::seconds(15))!=std::future_status::ready)throw std::runtime_error("Path result timeout");auto r=rf.get();
      if(r.code!=rclcpp_action::ResultCode::SUCCEEDED||r.result->path.poses.size()<2)throw std::runtime_error("Path unreachable");
      auto ps=r.result->path.poses;e.xy.resize(ps.size(),2);e.length=V::Zero(ps.size());
      for(size_t k=0;k<ps.size();k++){e.xy.row(k)<<ps[k].pose.position.x+15,ps[k].pose.position.y+10;if(k)e.length(k)=e.length(k-1)+(e.xy.row(k)-e.xy.row(k-1)).norm();}
      e.duration=travel_time(e.length(e.length.size()-1));
      graph.push_back({{"from",i},{"to",j},{"xy",array(e.xy)},{"duration_s",e.duration}});
    }
    save("graph.json",graph);Search search(gp,edges,cfg,"mi");initial_plan=search.run(0);save("initial_plan.json",initial_plan);
    if(initial_plan["timeout"].get<bool>()||initial_plan["route"].size()<2)throw std::runtime_error("Initial search incomplete");
    record("initial_plan_ready",{{"seconds",initial_plan["seconds"]}});ready=1;
  }catch(const std::exception&e){record("preparation_error",{{"message",e.what()}});ready=-1;}}
  // One replaceable latest-state slot, not one queued search per observation.
  // Only the ROS executor touches the live GP; the worker owns a frozen copy.
  void offer_latest(const std::string& reason){
    if(!active||returning||home_done)return;
    double t=now().seconds()-origin;
    int anchor=nav_active?target:current;
    double start=t;
    if(nav_active){
      // Project the current pose onto the planned polyline to estimate remaining arc length.
      const auto& edge=edges[current][anchor];
      Eigen::Vector2d location(x,y);
      double best_distance=INFINITY, along=0;
      for(int i=1;i<edge.xy.rows();++i){
        const Eigen::Vector2d a=edge.xy.row(i-1).transpose(), b=edge.xy.row(i).transpose();
        const Eigen::Vector2d segment=b-a;
        const double norm=segment.squaredNorm();
        if(norm<=0)continue;
        const double fraction=std::clamp((location-a).dot(segment)/norm,0.,1.);
        const double distance=(location-a-fraction*segment).squaredNorm();
        if(distance<best_distance){best_distance=distance;along=edge.length(i-1)+fraction*std::sqrt(norm);}
      }
      start+=travel_time(std::max(0.,edge.length(edge.length.size()-1)-along));
    }
    J state=gp.snapshot(); double sample_t=received?double(state["x"].back()[2]):0;
    J meta={{"sample_id_end",tick},{"gp_version",gp.version},{"assimilated_new_count",received},
      {"sample_t",sample_t},{"queued_t",t},{"queued_wall",wall()},{"anchor",anchor},{"goal_epoch",epoch},
      {"planning_start_t",start},{"weight_t",sample_t},{"reason",reason}};
    {std::lock_guard<std::mutex>l(queue_lock);
      if(!queue.empty()){queue.clear();coalesced++;}
      queue.push_back({std::move(state),std::move(meta)});peak_queue=std::max(peak_queue,queue.size());
    }
    condition.notify_one();
  }
  void update_gp(bool terminal){
    if(measurement_buffer.empty())return;
    J batch=measurement_buffer;measurement_buffer=J::array();double begin=wall();gp.append(batch);
    double update_s=wall()-begin;
    record("gp_update",{{"version",gp.version},{"seconds",update_s},{"batch_size",batch.size()},
      {"terminal",terminal},{"assimilated_new_count",received}});
    if(!terminal)offer_latest("new_gp");
  }
  void process_jobs(){
    while(true){
      Job job;
      {std::unique_lock<std::mutex>lock(queue_lock);condition.wait(lock,[this]{return stop_worker||!queue.empty();});if(stop_worker)break;job=std::move(queue.back());queue.clear();job.meta["id"]=++requested;}
      record("grg_start",job.meta);
      double begin=wall();J ans;
      try{
        GP frozen(job.state);double rebuild_s=wall()-begin;
        Search search(frozen,edges,cfg,"mi",job.meta["planning_start_t"],job.meta["weight_t"]);ans=search.run(job.meta["anchor"]);
        ans["gp_rebuild_s"]=rebuild_s;ans["total_compute_s"]=wall()-begin;ans["request"]=job.meta;
        ans["completion_t"]=now().seconds()-origin;ans["latency_s"]=wall()-double(job.meta["queued_wall"]);
        ans["queue_wait_s"]=begin-double(job.meta["queued_wall"]);ans["finished_after_window"]=stop_worker.load();
        save("plan_"+std::to_string(int(job.meta["id"]))+".json",ans);
        {std::lock_guard<std::mutex>l(result_lock);results[int(job.meta["goal_epoch"])]=ans;}
        completed++;record("grg_complete",{{"id",job.meta["id"]},{"gp_version",job.meta["gp_version"]},{"seconds",ans["total_compute_s"]},{"latency_s",ans["latency_s"]},{"timeout",ans["timeout"]}});
      }catch(const std::exception&e){record("grg_error",{{"id",job.meta["id"]},{"message",e.what()}});}
    }
    J rest=J::array();{std::lock_guard<std::mutex>l(queue_lock);for(const auto&job:queue)rest.push_back(job.meta);}
    // A pending state is a coalesced signal, not an accepted computation request.
    save("pending_state_at_stop.json",rest);save("unprocessed_requests.json",J::array());
    save("worker_status.json",{{"completed_total",completed.load()},{"unprocessed",0},
      {"peak_queue",peak_queue},{"coalesced_state_signals",coalesced},{"pending_state_at_stop",rest.size()}});
  }
  void dispatch(int id,int plan_id){
    if((id==Search::home)!=returning)throw std::runtime_error("Home is reserved for final return");
    if(nav_active)throw std::runtime_error("Attempt to switch a moving goal");
    // Preserve only the unexecuted sampling tail.
    if(retained_route.size()>1&&retained_route[0]==current&&retained_route[1]==id)
      retained_route.erase(retained_route.begin());
    else retained_route={id};
    target=id;nav_active=true;waiting=false;cancel_requested=false;goal_handle.reset();int token=++epoch;
    if(returning){std::lock_guard<std::mutex>l(queue_lock);queue.clear();}
    else offer_latest("goal_context");
    Nav::Goal g;g.pose=point(id,current);double a=std::atan2(points(id,1)-y,points(id,0)-x);g.pose.pose.orientation.z=std::sin(a/2);g.pose.pose.orientation.w=std::cos(a/2);
    rclcpp_action::Client<Nav>::SendGoalOptions opts;
    opts.goal_response_callback=[this,token](auto h){
      if(token!=epoch||!active)return;
      if(!h){
        nav_active=false;
        if(!sampling_stop_reason.empty()&&!returning)return;
        nav_failures++;finish("goal_rejected");return;
      }
      goal_handle=h;
      if(!sampling_stop_reason.empty()&&!returning)cancel_sampling_goal();
    };
    opts.result_callback=[this,token,id](const auto&r){
      if(token!=epoch||!active)return;
      nav_active=false;goal_handle.reset();
      if(!sampling_stop_reason.empty()&&!returning){
        record("sampling_goal_ended",{{"target",id},{"goal_epoch",token}});
        return;
      }
      if(r.code!=rclcpp_action::ResultCode::SUCCEEDED){
        nav_failures++;finish("navigation_failed");return;
      }
      current=id;arrived_t=now().seconds()-origin;
      if(returning){
        home_done=true;home_arrival_t=arrived_t;
        home_error_at_arrival=std::hypot(x-points(0,0),y-points(0,1));
        record("arrived",{{"target",id},{"goal_epoch",token}});
        record("home_returned",{{"home_arrival_t",home_arrival_t}});
        return;
      }
      segments++;waiting=true;
      record("arrived",{{"target",id},{"goal_epoch",token}});
    };
    record("nav_goal",{{"target",id},{"goal_epoch",token},{"selected_plan_id",plan_id},{"returning",returning}});nav->async_send_goal(g,opts);
  }
  double budget() const {return cfg["navigation_budget_s"];}
  bool can_sample_next(int id,double t) const {
    if(id==current||id<=Search::home||id>=points.rows())return false;
    double duration=edges[current][id].duration;
    double onward_return=edges[id][Search::home].duration;
    return std::isfinite(duration)&&duration>0&&t+duration+onward_return<=budget()+1e-8;
  }
  bool has_sampling_leg(double t) const {
    for(int id=Search::home+1;id<points.rows();++id)if(can_sample_next(id,t))return true;
    return false;
  }
  void cancel_sampling_goal(){
    if(!nav_active||!goal_handle||cancel_requested)return;
    cancel_requested=true;
    nav->async_cancel_goal(goal_handle);
    record("sampling_goal_cancel_requested",{{"target",target},{"goal_epoch",epoch}});
  }
  void end_sampling(const std::string&reason,double t){
    sampling_stop_reason=reason;sampling_end=std::min(t,budget());sampling_wall=wall()-started_wall;
    sampling_segments=segments;sampling_stop_node=current;stop_had_feasible_leg=has_sampling_leg(t);
    {std::lock_guard<std::mutex>l(queue_lock);
      stop_worker=true;completed_at_sampling_end=completed.load();started_at_sampling_end=requested.load();
    }
    condition.notify_all();
    if(waiting)waiting_s+=t-arrived_t;
    waiting=false;
    record("sampling_finished",{{"reason",reason},{"sampling_duration_s",sampling_end},
      {"last_waypoint",current},{"has_feasible_next_leg",stop_had_feasible_leg}});
    cancel_sampling_goal();
  }
  void finish(const std::string&status){
    if(finished)return;
    active=false;finished=true;stop_worker=true;condition.notify_all();
    if(nav_active)nav->async_cancel_all_goals();
    double stop_t=home_done?home_arrival_t:(origin?now().seconds()-origin:0),stop_wall=origin?wall()-started_wall:0;
    update_gp(true);
    save("final_gp.json",gp.snapshot());
    save("mission.json",{{"status",status},{"origin",origin},{"duration_s",stop_t},{"wall_s",stop_wall},
      {"mission_budget_s",budget()},{"sampling_duration_s",sampling_end},{"sampling_wall_s",sampling_wall},
      {"home_arrival_t",home_arrival_t},{"within_mission_budget",home_done&&home_arrival_t<=budget()},
      {"return_measurements",return_measurements},
      {"sampling_stop_reason",sampling_stop_reason},{"sampling_stop_node",sampling_stop_node},
      {"return_duration_s",return_start>=0?(home_done?home_arrival_t:stop_t)-return_start:0},
      {"return_started_t",return_start},{"return_navigation_wall_s",return_start>=0?wall()-return_wall:0},
      {"update_interval_s",cfg["update_interval_s"]},{"measurements",received},{"gp_updates",gp.version},
      {"grg_requested",started_at_sampling_end},{"grg_completed_at_stop",completed_at_sampling_end},
      {"missed_ticks",missed},{"stale_pose",stale},{"pending_sensor_requests",pending},{"segments",sampling_segments},
      {"navigation_failures",nav_failures},{"home_returned",home_done},
      {"home_error_m",home_done?home_error_at_arrival:std::hypot(x-points(0,0),y-points(0,1))},
      {"arrival_wait_s",waiting_s},{"goal_preemptions_before_deadline",0}});
    record("mission_finished",{{"status",status}});
  }
  void poll(){
    if(finished)return;
    if(!preparing&&now().seconds()>1&&paths->action_server_is_ready()&&nav->action_server_is_ready()&&query->get_subscription_count()>0&&now().seconds()-pose_time<.5){preparing=true;preparation=std::thread([this]{prepare();});}
    if(ready==-1){finish("preparation_failed");return;}
    if(!active&&ready==1){if(preparation.joinable())preparation.join();ready=2;origin=now().seconds();started_wall=wall();active=true;calculation=std::thread([this]{process_jobs();});record("mission_start");retained_route=initial_plan["route"].get<std::vector<int>>();dispatch(initial_plan["route"][1],0);}
    if(!active)return;
    double t=now().seconds()-origin;
    if(sampling_stop_reason.empty()&&t>=budget())end_sampling("budget_reached",t);
    if(!sampling_stop_reason.empty()){
      if(home_done&&pending==0){finish("window_completed");return;}
      if(t-sampling_end>180){finish("return_timeout");return;}
      if(!returning&&!nav_active){
        returning=true;return_start=t;return_wall=wall();
        record("return_started");
        dispatch(Search::home,-1);
      }
    }
    int due=int(std::floor(t+1e-9));
    if(!home_done&&due>tick){
      missed+=due-tick-1;tick=due;
      if(now().seconds()-pose_time>.5){stale++;}
      else{
        J row={{"id",tick},{"t",t},{"x",x},{"y",y},{"returning",returning},
          {"request_wall",wall()},{"pose_age_s",now().seconds()-pose_time}};
        Msg msg;msg.data=row.dump();pending++;query->publish(msg);record("query",row);
      }
    }
    if(!sampling_stop_reason.empty()||!waiting||home_done)return;
    if(!has_sampling_leg(t)){end_sampling("no_sampling_leg_fits",t);return;}
    J candidate;
    {std::lock_guard<std::mutex>l(result_lock);auto it=results.find(epoch);if(it!=results.end())candidate=it->second;}
    bool adopted=false;
    if(!candidate.is_null()&&!candidate["timeout"].get<bool>()){
      auto route=candidate["route"].get<std::vector<int>>();
      if(route.size()>1&&route[0]==current&&can_sample_next(route[1],t)){
        retained_route=route;retained_plan_id=candidate["request"]["id"];adopted=true;
      }
    }
    if(retained_route.size()<2||retained_route[0]!=current)return;
    int next=retained_route[1];
    if(!can_sample_next(next,t))return;
    waiting_s+=t-arrived_t;
    J selection={{"selected_plan_id",retained_plan_id},{"route_source",adopted?"fresh_completed":"retained_tail"},
      {"remaining_route",retained_route},{"next",next}};
    if(adopted){selection["plan_age_s"]=t-double(candidate["request"]["sample_t"]);
      selection["gp_version_lag"]=gp.version-int(candidate["request"]["gp_version"]);}
    record("arrival_selection",selection);dispatch(next,retained_plan_id);
  }
};
int main(int argc,char**argv){auto threads=configure_matrix_threads();save("matrix_threads.json",threads);rclcpp::init(argc,argv);rclcpp::spin(std::make_shared<Experiment>());rclcpp::shutdown();}
