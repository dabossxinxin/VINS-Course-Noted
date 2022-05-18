cmake_minimum_required(VERSION 2.8.3)
project(vins_estimator)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++17")
set(CMAKE_CXX_FLAGS "-bigobj")
set(CMAKE_CXX_FLAGS "-Dprotobuf_MSVC_STATIC_RUNTIME=OFF ")
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

add_definitions(-DGLOG_NO_ABBREVIATED_SEVERITIES)

#find_package(Eigen3 REQUIRED)
#find_package(Ceres REQUIRED)
#find_package(Pangolin REQUIRED)
#find_package(OpenCV REQUIRED)
#find_package(Boost REQUIRED COMPONENTS filesystem program_options system)

include(FindEigen3)
include(FindCeres)
include(FindGLOG)
include(FindOpenCV)
include(FindBoost)
include(FindPangolin)
include(FindGlew)

#include_directories(
#  ${PROJECT_SOURCE_DIR}/include
#  ${EIGEN3_INCLUDE_DIR}
#  ${CERES_INCLUDE_DIRS}
#  ${Pangolin_INCLUDE_DIRS}
#  ${Boost_INCLUDE_DIRS}
#)
include_directories(${Boost_INCLUDE_DIRS})
include_directories(${EIGEN3_INCLUDE_DIR})
include_directories(${GFLAGS_INCLUDE_DIRS})
include_directories(${GLOG_INCLUDE_DIRS})
include_directories(${CERES_INCLUDE_DIRS})
include_directories(${OpenCV3_INCLUDE_DIR})
include_directories(${Pangolin_INCLUDE_DIRS})
include_directories(${GLEW_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/include)

#add_library(camera_model STATIC
#    src/camera_models/chessboard/Chessboard.cc
#    src/camera_models/calib/CameraCalibration.cc
#    src/camera_models/camera_models/Camera.cc
#    src/camera_models/camera_models/CameraFactory.cc
#    src/camera_models/camera_models/CostFunctionFactory.cc
#    src/camera_models/camera_models/PinholeCamera.cc
#    src/camera_models/camera_models/CataCamera.cc
#    src/camera_models/camera_models/EquidistantCamera.cc
#    src/camera_models/camera_models/ScaramuzzaCamera.cc
#    src/camera_models/sparse_graph/Transform.cc
#    src/camera_models/gpl/gpl.cc
#    src/camera_models/gpl/EigenQuaternionParameterization.cc)

#target_link_libraries(camera_model debug ${OpenCV3_DEBUG_LIB}	optimized ${OpenCV3_RELEASE_LIB})
#target_link_libraries(camera_model debug ${GLOG_DEBUG_LIB} 	optimized ${GLOG_RELEASE_LIB})
#target_link_libraries(camera_model debug ${CERES_DEBUG_LIB} 	optimized ${CERES_RELEASE_LIB})
#FOREACH(_var ${Boost_LIBRARIES})
#	 TARGET_LINK_LIBRARIES(camera_model debug ${Boost_LIB_DIR}/${_var}-gd-x64-1_68.lib
#	optimized ${Boost_LIB_DIR}/${_var}-x64-1_68.lib)
#ENDFOREACH()
#foreach(_var ${GFLAGS_LIBRARIES})
#    target_link_libraries(camera_model debug ${GFLAGS_LIB_DIR}/${_var}_debug.lib
#	optimized ${GFLAGS_LIB_DIR}/${_var}.lib)
#endforeach()

ADD_LIBRARY(MyVio STATIC
    src/System.cpp
    src/parameters.cpp
    src/estimator.cpp
    src/feature_manager.cpp
    src/feature_tracker.cpp

    src/utility/utility.cpp
    src/initial/solve_5pts.cpp
    src/initial/initial_aligment.cpp
    src/initial/initial_sfm.cpp
    src/initial/initial_ex_rotation.cpp

    src/backend/vertex.cpp
    src/backend/edge.cpp
    src/backend/problem.cpp
    src/backend/vertex_pose.cpp
    src/backend/edge_reprojection.cpp
    src/backend/edge_imu.cpp
    src/backend/edge_prior.cpp
    src/backend/loss_function.cpp
    src/backend/imu_integration.cpp
    )

#target_link_libraries(MyVio  
#  ${OpenCV_LIBS}  
#  ${CERES_LIBRARIES}
#  ${Pangolin_LIBRARIES}
#  camera_model)
#target_link_libraries(MyVio camera_model)
#target_link_libraries(${Pangolin_DEBUG_LIB} ${GLEW_DEBUG_LIB})
#target_link_libraries(${Pangolin_RELEASE_LIB} ${GLEW_RELEASE_LIB})
target_link_libraries(MyVio opengl32.lib)
target_link_libraries(MyVio debug ${GLEW_DEBUG_LIB} 	optimized ${GLEW_RELEASE_LIB})
target_link_libraries(MyVio debug ${Pangolin_DEBUG_LIB} optimized ${Pangolin_RELEASE_LIB})
target_link_libraries(MyVio debug ${OpenCV3_DEBUG_LIB}	optimized ${OpenCV3_RELEASE_LIB})
target_link_libraries(MyVio debug ${CERES_DEBUG_LIB} 	optimized ${CERES_RELEASE_LIB})


add_executable(run_euroc test/run_euroc.cpp)
target_link_libraries(run_euroc 
  MyVio  
  -lpthread) 

add_executable(testCurveFitting test/CurveFitting.cpp)
target_link_libraries(testCurveFitting MyVio)

