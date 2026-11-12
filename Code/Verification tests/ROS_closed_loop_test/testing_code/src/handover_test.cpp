#include "handover.hpp"
#include <cassert>
#include <iostream>
int main() {
  Handover s;
  auto id = s.begin_request(0);
  assert(s.accept(id, 0));
  s.depart(29);
  id = s.begin_request(29);
  assert(!s.accept(id - 1, 29));
  assert(!s.accept(id, 3));
  assert(s.accept(id, 29));
  assert(s.moving);
  s.arrive(29);
  id = s.begin_request(29);
  assert(s.accept(id, 29));
  s.depart(3);
  id = s.begin_request(3);
  s.arrive(3);
  assert(s.outstanding);
  assert(s.accept(id, 3));
  id = s.begin_request(3);
  assert(s.accept(id, 3));
  s.depart(13);
  bool rejected = false;
  try {
    s.depart(0);
  } catch (...) {
    rejected = true;
  }
  assert(rejected);
  s.arrive(13);
  id = s.begin_request(13);
  assert(!s.accept(id, 0));
  assert(s.accept(id, 13));
  std::cout << "PASS: early/late replies; stale IDs; wrong anchors; no goal "
               "preemption."
            << std::endl;
}
