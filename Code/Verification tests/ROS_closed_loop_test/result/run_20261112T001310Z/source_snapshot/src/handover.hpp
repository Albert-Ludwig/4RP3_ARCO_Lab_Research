#pragma once
#include <stdexcept>
// A candidate may be activated only at its physical anchor.
// Identity checks reject delayed replies and prevent goal preemption.
struct Handover {
  int current = 0, target = -1, request = 0, anchor = -1;
  bool moving = false, outstanding = false;
  int begin_request(int from) {
    if (outstanding)
      throw std::runtime_error("Concurrent planning request");
    anchor = from;
    outstanding = true;
    return ++request;
  }
  bool accept(int id, int from) {
    if (!outstanding || id != request || from != anchor ||
        from != (moving ? target : current))
      return false;
    outstanding = false;
    return true;
  }
  void depart(int next) {
    if (moving || outstanding || next == current)
      throw std::runtime_error("Invalid goal handover");
    target = next;
    moving = true;
  }
  void arrive(int next) {
    if (!moving || next != target)
      throw std::runtime_error("Unexpected arrival");
    current = next;
    moving = false;
  }
};
