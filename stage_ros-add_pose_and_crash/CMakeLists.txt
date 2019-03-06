cmake_minimum_required(VERSION 2.8.3)
project(stage_ros_add_pose_and_crash)

find_package(catkin REQUIRED
  COMPONENTS
    geometry_msgs
    nav_msgs
    roscpp
    sensor_msgs
    std_msgs
    std_srvs
    tf
)

find_package(Boost REQUIRED COMPONENTS system thread)

find_package(stage REQUIRED)

include_directories(
  ${catkin_INCLUDE_DIRS}
  ${Boost_INCLUDE_DIRS}
  ${STAGE_INCLUDE_DIRS}
)

catkin_package()

# Declare a cpp executable
add_executable(stageros src/stageros.cpp)
set(${PROJECT_NAME}_extra_libs "")
if(UNIX AND NOT APPLE)
  set(${PROJECT_NAME}_extra_libs dl)
endif()
target_link_libraries(stageros
  ${catkin_LIBRARIES}
  ${Boost_LIBRARIES}
  ${STAGE_LIBRARIES}
  ${${PROJECT_NAME}_extra_libs}
)
if(catkin_EXPORTED_TARGETS)
  add_dependencies(stageros ${catkin_EXPORTED_TARGETS})
endif()

## Install

install(PROGRAMS scripts/upgrade-world.sh
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})

install(TARGETS stageros
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY rviz
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})

install(DIRECTORY world
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})

## Tests

if(CATKIN_ENABLE_TESTING)
  find_package(rostest REQUIRED)
  add_rostest(test/hztest.xml)
  add_rostest(test/cmdpose_tests.xml)
endif()
