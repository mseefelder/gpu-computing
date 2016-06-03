cmake_minimum_required(VERSION 2.8.12)
set(PROJECT_NAME "tracker")
project(${PROJECT_NAME} C CXX)

set(TRACKER_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
include_directories(${TRACKER_DIR})

#SET THREADING
find_package (Threads REQUIRED)

#SET OPENCV
#find_package( OpenCV REQUIRED )

#SET GLEW
find_package(GLEW REQUIRED)
if(NOT GLEW_FOUND)
  message(SEND_ERROR "GLEW not found on your system.")
endif()

#SET OPENGL
find_package(OpenGL REQUIRED)
if(NOT OPENGL_FOUND)
  message(SEND_ERROR "OpenGL not found on your system.")
endif()

#SET OPENMP
find_package(OpenMP REQUIRED)
if(NOT OPENMP_FOUND)
  message(SEND_ERROR "OpenMP not found on your system.")
else()
  set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

#SET GLFW
if( WIN32 ) # true if windows (32 and 64 bit)
  set (GLFW_INCLUDE_DIR "NOT-FOUND" CACHE PATH "glfw include directory")
  set (GLFW_LIBRARY_DIR "NOT-FOUND" CACHE PATH "glfw library directory")
  include_directories (${GLFW_INCLUDE_DIR})
  link_directories  (${GLFW_LIBRARY_DIR})
  set(GLFW_LIBRARIES glfw3)
else()
  find_package(PkgConfig)
  if(NOT PKG_CONFIG_FOUND)
    set (GLFW_INCLUDE_DIR "NOT-FOUND" CACHE PATH "glfw include directory")
    set (GLFW_LIBRARY_DIR "NOT-FOUND" CACHE PATH "glfw library directory")
    include_directories (${GLFW_INCLUDE_DIR})
    link_directories  (${GLFW_LIBRARY_DIR})
    set(GLFW_LIBRARIES glfw3)
  else()
    pkg_search_module(GLFW REQUIRED glfw3)
    set(GLFW_LIBRARIES ${GLFW_STATIC_LIBRARIES})
  endif()
endif()

#SET EIGEN3
if( WIN32 ) # true if windows (32 and 64 bit)
  set (EIGEN3_INCLUDE_DIRS "NOT-FOUND" CACHE PATH "eigen3 include directory")
  if( NOT EIGEN3_INCLUDE_DIRS )
    message( SEND_ERROR "Point EIGEN3_INCLUDE_DIR to the include directory of your Eigen3 installation.")
  endif()
else()
  find_package(PkgConfig)
  if(NOT PKG_CONFIG_FOUND)
    set (EIGEN3_INCLUDE_DIRS "NOT-FOUND" CACHE PATH "eigen3 include directory")
    if( NOT EIGEN3_INCLUDE_DIRS )
      message( SEND_ERROR "Point EIGEN3_INCLUDE_DIR to the include directory of your Eigen3 installation.")
    endif()
  else()
    pkg_check_modules(EIGEN3 REQUIRED eigen3)
  endif()
endif()

#SET TUCANO
set(TUCANO_ROOT "NOT-FOUND" CACHE PATH "Please, set this path to Tucano root.")
if( TUCANO_ROOT )
  set(TUCANO_EFFECTS_DIR "${TUCANO_ROOT}/effects")
  set(TUCANO_SRC_DIR "${TUCANO_ROOT}/src")
  set(
    TUCANO_SOURCES
    ${TUCANO_SRC_DIR}/tucano.hpp
    )
  set(
    TUCANO_SOURCES_UTILS
    ${TUCANO_SRC_DIR}/utils/misc.hpp
    ${TUCANO_SRC_DIR}/utils/trackball.hpp
    ${TUCANO_SRC_DIR}/utils/objimporter.hpp
    ${TUCANO_SRC_DIR}/utils/plyimporter.hpp
    ${TUCANO_SRC_DIR}/utils/qtplainwidget.hpp
    ${TUCANO_SRC_DIR}/utils/qttrackballwidget.hpp
    ${TUCANO_SRC_DIR}/utils/rply.hpp
    )
else( TUCANO_ROOT )
  message( SEND_ERROR "Please, point TUCANO_ROOT to Tucano root directory.")
endif( TUCANO_ROOT )

#WHERE TO STORE EXECUTABLE
set(TRACKER_BINARY_DIR "${PROJECT_SOURCE_DIR}/bin")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${TRACKER_BINARY_DIR})
foreach( OUTPUTCONFIG ${CMAKE_CONFIGURATION_TYPES} )
    string( TOUPPER ${OUTPUTCONFIG} OUTPUTCONFIG )
    set( CMAKE_RUNTIME_OUTPUT_DIRECTORY_${OUTPUTCONFIG} ${TRACKER_BINARY_DIR} )
endforeach( OUTPUTCONFIG CMAKE_CONFIGURATION_TYPES )

#BUILD PROJECT
include_directories(
  ${GLEW_INCLUDE_DIRS}
  ${OPENGL_INCLUDE_DIR}
  ${EIGEN3_INCLUDE_DIRS}
  ${TUCANO_EFFECTS_DIR}
  ${TUCANO_SRC_DIR}
  ${OpenCV_INCLUDE_DIRS}
  )

set(
  TRACKER_SOURCES
  ${TRACKER_DIR}/main.cpp
  )

add_executable(
  tracker
  ${TUCANO_SOURCES}
  ${TUCANO_SOURCES_UTILS}
  ${TRACKER_SOURCES}
  )

set_property(
  TARGET tracker
  PROPERTY COMPILE_FLAGS
  "-DTUCANOSHADERDIR=${TUCANO_EFFECTS_DIR}/shaders/ -g -pthread -std=c++11"
  )

target_link_libraries(
  tracker
  ${CMAKE_THREAD_LIBS_INIT}
  ${GLEW_LIBRARIES}
  ${GLFW_LIBRARIES}
  ${OPENGL_LIBRARIES}
  ${OpenCV_LIBS}
  )
