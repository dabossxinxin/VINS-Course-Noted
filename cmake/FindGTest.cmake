
FIND_PATH(GTest_INCLUDE_DIRS GL HINTS ${CMAKE_SOURCE_DIR}/../../gtest/include)
FIND_PATH(GTest_LIB_DIR glew.lib HINTS ${CMAKE_SOURCE_DIR}/../../gtest/lib)

SET(GTest_LIBRARIES gtest gtest_main)