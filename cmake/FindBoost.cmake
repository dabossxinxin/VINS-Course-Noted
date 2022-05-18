
SET(USE_Boost_VERSION "1.68" CACHE STRING "Expected Boost version")

SET_PROPERTY(CACHE USE_Boost_VERSION PROPERTY STRINGS 1.68)

#IF(MSVC_VERSION EQUAL 1900)
#    IF(USE_VTK_VERSION EQUAL 8.1)
#        FIND_PATH(VTK_INCLUDE_DIR NAMES vtk-8.1 HINTS ${CMAKE_SOURCE_DIR}/../../SDK/VTK_8.1.2/include)
#        FIND_PATH(VTK_LIB_DIR NAMES vtkCommonColor-8.1.lib HINTS ${CMAKE_SOURCE_DIR}/../../SDK/VTK_8.1.2/lib)
#    ENDIF()
#ENDIF()

IF(USE_Boost_VERSION EQUAL 1.68)
	FIND_PATH(Boost_INCLUDE_DIR NAMES boost-1_68 HINTS ${CMAKE_SOURCE_DIR}/../../SDK/Boost/include)
	FIND_PATH(Boost_LIB_DIR NAMES libboost_atomic-vc140-mt-gd-x64-1_68.lib HINTS ${CMAKE_SOURCE_DIR}/../../SDK/Boost/lib)
ENDIF()

IF(USE_Boost_VERSION EQUAL 1.68)
  SET(Boost_LIBRARIES
	libboost_atomic-vc140-mt
	libboost_chrono-vc140-mt
	libboost_container-vc140-mt
	libboost_context-vc140-mt
	libboost_contract-vc140-mt
	libboost_coroutine-vc140-mt
	libboost_date_time-vc140-mt
	libboost_exception-vc140-mt
	libboost_fiber-vc140-mt
	libboost_filesystem-vc140-mt
	libboost_graph-vc140-mt
	libboost_iostreams-vc140-mt
	libboost_locale-vc140-mt
	libboost_log-vc140-mt
	libboost_log_setup-vc140-mt
	libboost_math_c99-vc140-mt
	libboost_math_c99f-vc140-mt
	libboost_math_c99l-vc140-mt
	libboost_math_tr1-vc140-mt
	libboost_math_tr1f-vc140-mt
	libboost_math_tr1l-vc140-mt
	libboost_prg_exec_monitor-vc140-mt
	libboost_program_options-vc140-mt
	libboost_random-vc140-mt
	libboost_regex-vc140-mt
	libboost_serialization-vc140-mt
	libboost_signals-vc140-mt
	libboost_stacktrace_noop-vc140-mt
	libboost_stacktrace_windbg-vc140-mt
	libboost_stacktrace_windbg_cached-vc140-mt
	libboost_system-vc140-mt
	libboost_test_exec_monitor-vc140-mt
	libboost_thread-vc140-mt
	libboost_timer-vc140-mt
	libboost_type_erasure-vc140-mt
	libboost_unit_test_framework-vc140-mt
	libboost_wave-vc140-mt
	libboost_wserialization-vc140-mt
)

  SET(Boost_INCLUDE_DIRS ${Boost_INCLUDE_DIR}/boost-1_68)
ENDIF()
