cmake_minimum_required(VERSION 3.8.3)
project(vins_estimator)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++11")
set(CMAKE_CXX_FLAGS "-Dprotobuf_MSVC_STATIC_RUNTIME=OFF")
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

add_definitions(-DUSE_OPENMP)
add_definitions(-DGLOG_NO_ABBREVIATED_SEVERITIES)

include(FindGlew)
include(FindGLOG)
include(FindCeres)
include(FindBoost)
include(FindEigen3)
include(FindOpenCV)
include(FindPangolin)

include_directories(${GLOG_INCLUDE_DIRS})
include_directories(${GLEW_INCLUDE_DIRS})
include_directories(${Boost_INCLUDE_DIRS})
include_directories(${EIGEN3_INCLUDE_DIR})
include_directories(${GFLAGS_INCLUDE_DIRS})
include_directories(${CERES_INCLUDE_DIRS})
include_directories(${OpenCV3_INCLUDE_DIR})
include_directories(${Pangolin_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/include)

########################################################
#													   #
#				  ±‡“Îcamera modelø‚			       #
#													   #	
########################################################
file(GLOB CalibHdrs			"./include/camera_models/calib/*.h")
file(GLOB CameraHdrs		"./include/camera_models/camera_models/*.h")
file(GLOB ChessBoardHdrs	"./include/camera_models/chessboard/*.h")
file(GLOB GplHdrs			"./include/camera_models/gpl/*.h")
file(GLOB SparseGraphHdrs	"./include/camera_models/sparse_graph/*.h")

file(GLOB CalibSrcs			"./src/camera_models/calib/*.cpp")
file(GLOB CameraSrcs		"./src/camera_models/camera_models/*.cpp")
file(GLOB ChessBoardSrcs	"./src/camera_models/chessboard/*.cpp")
file(GLOB GplSrcs			"./src/camera_models/gpl/*.cpp")
file(GLOB SparseGraphSrcs	"./src/camera_models/sparse_graph/*.cpp")

source_group("Calib Srcs"		FILES ${CalibSrcs})
source_group("Calib Hdrs"		FILES ${CalibHdrs})
source_group("Camera Srcs"		FILES ${CameraSrcs})
source_group("Camera Hdrs"		FILES ${CameraHdrs})
source_group("ChessBoard Srcs"	FILES ${ChessBoardSrcs})
source_group("ChessBoard Hdrs"	FILES ${ChessBoardHdrs})
source_group("Gpl Srcs"			FILES ${GplSrcs})
source_group("Gpl Hdrs"			FILES ${GplHdrs})
source_group("SparseGraph Srcs" FILES ${SparseGraphSrcs})
source_group("SparseGraph Hdrs" FILES ${SparseGraphHdrs})

add_library(CameraModel STATIC
    ${CalibSrcs}
	${CalibHdrs}
	${CameraSrcs}
	${CameraHdrs}
	${ChessBoardSrcs}
	${ChessBoardHdrs}
	${GplSrcs}
	${GplHdrs}
	${SparseGraphSrcs}
	${SparseGraphHdrs}
)
set_target_properties(CameraModel PROPERTIES COMPILE_FLAGS "/EHsc")
target_compile_options(CameraModel PRIVATE $<$<CXX_COMPILER_ID:MSVC>:/bigobj>)

target_link_libraries(CameraModel debug ${OpenCV3_DEBUG_LIB}	optimized ${OpenCV3_RELEASE_LIB})
target_link_libraries(CameraModel debug ${GLOG_DEBUG_LIB} 		optimized ${GLOG_RELEASE_LIB})
target_link_libraries(CameraModel debug ${CERES_DEBUG_LIB} 		optimized ${CERES_RELEASE_LIB})
FOREACH(_var ${Boost_LIBRARIES})
	 TARGET_LINK_LIBRARIES(CameraModel debug ${Boost_LIB_DIR}/${_var}-gd-x64-1_68.lib
									optimized ${Boost_LIB_DIR}/${_var}-x64-1_68.lib)
ENDFOREACH()
foreach(_var ${GFLAGS_LIBRARIES})
    target_link_libraries(CameraModel debug ${GFLAGS_LIB_DIR}/${_var}_debug.lib
									optimized ${GFLAGS_LIB_DIR}/${_var}.lib)
endforeach()

########################################################
#													   #
#			±‡“Îvisual inertial odometryø‚			   #
#													   #	
########################################################
file(GLOB VioHdrs			"./include/*.h")
file(GLOB VioFactorHdrs		"./include/factor/*.h")
file(GLOB VioUtilityHdrs	"./include/utility/*.h")
file(GLOB VioInitialHdrs	"./include/initial/*.h")
file(GLOB VioBackEndHdrs	"./include/backend/*.h")

file(GLOB VioSrcs			"./src/*.cpp")
file(GLOB VioFactorSrcs		"./src/factor/*.cpp")
file(GLOB VioUtilitySrcs	"./src/utility/*.cpp")
file(GLOB VioInitialSrcs	"./src/initial/*.cpp")
file(GLOB VioBackEndSrcs	"./src/backend/*.cpp")

source_group("Main Srcs"	FILES ${VioSrcs})
source_group("Main Hdrs"	FILES ${VioHdrs})
source_group("Factor Srcs"	FILES ${VioFactorSrcs})
source_group("Factor Hdrs"	FILES ${VioFactorHdrs})
source_group("Utility Srcs" FILES ${VioUtilitySrcs})
source_group("Utility Hdrs" FILES ${VioUtilityHdrs})
source_group("Initial Srcs" FILES ${VioInitialSrcs})
source_group("Initial Hdrs" FILES ${VioInitialHdrs})
source_group("BackEnd Srcs" FILES ${VioBackEndSrcs})
source_group("BackEnd Hdrs" FILES ${VioBackEndHdrs})

ADD_LIBRARY(VIOdometry STATIC 
	${VioHdrs} 
	${VioSrcs}
	${VioFactorSrcs}
	${VioFactorHdrs}
	${VioUtilitySrcs}
	${VioUtilityHdrs}
	${VioInitialSrcs}
	${VioInitialHdrs}
	${VioBackEndSrcs}
	${VioBackEndHdrs}
)
set_target_properties(VIOdometry PROPERTIES COMPILE_FLAGS "/EHsc")

target_link_libraries(VIOdometry CameraModel)
target_link_libraries(VIOdometry opengl32.lib)
target_link_libraries(VIOdometry debug ${GLEW_DEBUG_LIB} 		optimized ${GLEW_RELEASE_LIB})
target_link_libraries(VIOdometry debug ${Pangolin_DEBUG_LIB}	optimized ${Pangolin_RELEASE_LIB})
target_link_libraries(VIOdometry debug ${OpenCV3_DEBUG_LIB}		optimized ${OpenCV3_RELEASE_LIB})
target_link_libraries(VIOdometry debug ${CERES_DEBUG_LIB} 		optimized ${CERES_RELEASE_LIB})

########################################################
#													   #	
#					±‡“Îø…÷¥––Œƒº˛			           #
#													   #	
########################################################
add_executable(RunEuroc example/run_euroc.cpp)
set_target_properties(RunEuroc PROPERTIES COMPILE_FLAGS "/EHsc")
target_link_libraries(RunEuroc VIOdometry -lpthread) 

add_executable(CurveFitting example/CurveFitting.cpp)
set_target_properties(CurveFitting PROPERTIES COMPILE_FLAGS "/EHsc")
target_link_libraries(CurveFitting VIOdometry)

add_executable(MonoOptimization example/MonoOptimization.cpp)
target_link_libraries(MonoOptimization VIOdometry)