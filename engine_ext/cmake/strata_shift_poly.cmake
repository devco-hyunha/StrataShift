# engine_ext/cmake/strata_shift_poly.cmake - StrataShift's file (engine_ext/NOTICE.md), not PolyStrata's.
#
# Included by the copy of the pinned CMakeLists.txt right after the strata-poly target is defined
# (engine_ext/overlay.json).  strata-shift-poly is strata-poly with the architectures of engine_ext beside the
# engine's own: the same sources, include directories, libraries, definitions and options, read from the
# strata-poly target so that a pin which changes them is followed, plus src/poly/mixtral.cpp and STRATASHIFT_EXT,
# which turns on the dispatch lines engine_ext adds to engine.cpp.

foreach(_prop SOURCES INCLUDE_DIRECTORIES LINK_LIBRARIES COMPILE_DEFINITIONS COMPILE_OPTIONS)
  get_target_property(_strata_shift_${_prop} strata-poly ${_prop})
  if(NOT _strata_shift_${_prop})
    set(_strata_shift_${_prop} "")
  endif()
endforeach()

add_executable(strata-shift-poly ${_strata_shift_SOURCES} src/poly/mixtral.cpp)
target_include_directories(strata-shift-poly PRIVATE ${_strata_shift_INCLUDE_DIRECTORIES})
target_link_libraries(strata-shift-poly PRIVATE ${_strata_shift_LINK_LIBRARIES})
target_compile_definitions(strata-shift-poly PRIVATE ${_strata_shift_COMPILE_DEFINITIONS} STRATASHIFT_EXT=1)
target_compile_options(strata-shift-poly PRIVATE ${_strata_shift_COMPILE_OPTIONS})
