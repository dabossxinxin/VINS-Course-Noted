
FIND_PATH(Pangolin_INCLUDE_DIRS pangolin HINTS ${CMAKE_SOURCE_DIR}/../../Pangolin/include)
FIND_PATH(Pangolin_LIB_DIR pangolin.lib HINTS ${CMAKE_SOURCE_DIR}/../../Pangolin/lib)
SET(Pangolin_DEBUG_LIB ${Pangolin_LIB_DIR}/pangolind.lib)
SET(Pangolin_RELEASE_LIB ${Pangolin_LIB_DIR}/pangolin.lib)