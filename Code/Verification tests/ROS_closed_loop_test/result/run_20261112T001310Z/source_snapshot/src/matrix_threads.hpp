#pragma once
#include <Eigen/Core>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <nlohmann/json.hpp>
#include <omp.h>
// Configure before any GP construction. OMP and Eigen use the same bounded pool.
inline nlohmann::json configure_matrix_threads() {
  const char *value = std::getenv("CL_MATRIX_THREADS");
  int requested = value ? std::stoi(value) : 2;
  if (requested < 1 || requested > 4)
    throw std::runtime_error("CL_MATRIX_THREADS must be between 1 and 4");
  omp_set_dynamic(0);
  omp_set_num_threads(requested);
  Eigen::setNbThreads(requested);
  return {{"requested", requested}, {"eigen_effective", Eigen::nbThreads()},
          {"openmp_version", _OPENMP}, {"openmp_dynamic", omp_get_dynamic()},
          {"openmp_max_threads", omp_get_max_threads()}};
}
