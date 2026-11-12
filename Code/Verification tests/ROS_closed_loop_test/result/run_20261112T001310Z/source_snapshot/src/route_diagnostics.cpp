#include "core.hpp"
#include "matrix_threads.hpp"
#include <iostream>
J load(const std::string &name) {std::ifstream f(name);J j;f>>j;return j;}
std::vector<std::vector<Edge>> graph(const J &rows,int n) {
 std::vector<std::vector<Edge>> e(n,std::vector<Edge>(n));
 for(int i=0;i<n;++i){e[i][i].xy=M::Zero(2,2);e[i][i].length=V::Zero(2);e[i][i].duration=0;}
 for(const auto &r:rows){
  auto &a=e[int(r["from"])][int(r["to"])];a.xy=matrix(r["xy"]);a.duration=r["duration_s"];
  a.length=V::Zero(a.xy.rows());for(int k=1;k<a.xy.rows();++k)a.length(k)=a.length(k-1)+(a.xy.row(k)-a.xy.row(k-1)).norm();
 }
 return e;
}
J evaluate(Search &s,const M &events) {
 M z=s.gp.features(events),c=s.gp.kernel(s.q,events)-s.qfeatures.transpose()*z;
 M cov=s.gp.kernel(events,events)-z.transpose()*z;cov=(cov+cov.transpose()).eval()*.5;
 cov.diagonal().array()+=s.gp.noise;Eigen::LLT<M> f(cov);
 if(f.info()!=Eigen::Success)throw std::runtime_error("Diagnostic Cholesky failed");
 M solved=f.matrixL().solve(c.transpose());
 V reduction=solved.array().square().colwise().sum().transpose();
 V cells=V::Zero(70);for(int i=0;i<reduction.size();++i)cells(i%70)+=s.weights(i)*reduction(i);
 return {{"score",s.weights.dot(reduction)},{"spatial_contributions",std::vector<double>(cells.data(),cells.data()+70)},{"event_count",events.rows()}};
}
int main(int argc,char **argv){
 if(argc!=4)return 2;
 J proof=configure_matrix_threads();
 std::string root=argv[2];auto initial=load(root+"/initial.json");
 GP gp(initial);auto e=graph(load(root+"/graph.json"),31);
 double t=.132;J result{{"matrix_threads",proof},{"initial_length",initial["length"]},{"elapsed_s",t}};
 if(std::string(argv[1])=="complete"){
  double begin=wall();Search s(gp,e,t,300-t,2);s.deadline=INFINITY;
  auto answer=s.run(0);answer["complete_wall_s"]=wall()-begin;result["search"]=answer;
 }else{
  J routes={{"straight",{0,28,3,28}},{"previous_bent",{0,29,1,29}},
            {"lower_bent",{0,27,5,27}},{"vertical",{0,15,11,15,11,15}}};
  for(int swapped=0;swapped<=1;++swapped){
   if(swapped){std::swap(gp.length(0),gp.length(1));gp.factor();}
   Search s(gp,e,t,300-t,2);s.deadline=INFINITY;
   for(auto it=routes.begin();it!=routes.end();++it){
    auto route=it.value().get<std::vector<int>>();M events=s.events(route);
    if(events.rows()<180)throw std::runtime_error("Insufficient matched events");
    M matched=events.topRows(180);auto value=evaluate(s,matched);
    value["route"]=route;value["full_route_duration_s"]=s.finish(route)-t;
    value["events"]=array(matched);
    result[swapped?"axis_swapped_counterfactual":"fitted_kernel"][it.key()]=value;
   }
  }
  result["scope"]="Matched first 180 1 Hz events, same GP observations/initial time/grid/noise and frozen weighting rule. Axis swap is an untrained diagnostic counterfactual, not a deployed kernel or physical-field conclusion.";
 }
 std::ofstream(argv[3])<<result.dump(2);
 std::cout<<result.dump()<<std::endl;
}
