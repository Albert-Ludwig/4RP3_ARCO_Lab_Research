cmake_minimum_required(VERSION 3.8)
project(well_e_nav2_test)
find_package(ament_cmake REQUIRED)
install(DIRECTORY launch config DESTINATION share/${PROJECT_NAME})
install(PROGRAMS scripts/truth_localization.py scripts/navigation_probe.py
        DESTINATION lib/${PROJECT_NAME})
install(FILES scripts/geometry.py DESTINATION lib/${PROJECT_NAME})
ament_package()
