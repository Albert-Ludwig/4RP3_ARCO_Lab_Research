cmake_minimum_required(VERSION 3.8)
project(well_e_bunker_sim)
find_package(ament_cmake REQUIRED)
install(DIRECTORY launch config DESTINATION share/${PROJECT_NAME})
install(PROGRAMS scripts/cmd_watchdog.py scripts/odometry_adapter.py scripts/smoke_test.py
        scripts/turning_test.py scripts/steering_validation.py DESTINATION lib/${PROJECT_NAME})
ament_package()
