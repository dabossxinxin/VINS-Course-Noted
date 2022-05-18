
FIND_PATH(OpenCV3_INCLUDE_DIR NAMES opencv2 HINTS ${CMAKE_SOURCE_DIR}/../../SDK/opencv3.4.6/include)
FIND_FILE(OpenCV3_DEBUG_LIB opencv_world346d.lib HINTS ${CMAKE_SOURCE_DIR}/../../SDK/opencv-3.4.6/lib)
FIND_FILE(OpenCV3_RELEASE_LIB opencv_world346.lib HINTS ${CMAKE_SOURCE_DIR}/../../SDK/opencv-3.4.6/lib)
