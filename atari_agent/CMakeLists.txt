cmake_minimum_required(VERSION 3.14)

# ----------------------------------------------------------------------------
# Atari agent
# ----------------------------------------------------------------------------

project(atari_agent
	VERSION 0.1.0
	DESCRIPTION "Atari agent"
	LANGUAGES CXX
)

# ----------------------------------------------------------------------------
# Dependencies
# ----------------------------------------------------------------------------

include(${CMAKE_SOURCE_DIR}/cmake/drla.cmake)
include(${CMAKE_SOURCE_DIR}/cmake/spdlog.cmake)
include(${CMAKE_SOURCE_DIR}/cmake/ale.cmake)
include(${CMAKE_SOURCE_DIR}/cmake/nlohmann_json.cmake)

set(THREADS_PREFER_PTHREAD_FLAG ON)  # Set -pthreads
find_package(Threads REQUIRED)
find_package(Torch REQUIRED)

# ----------------------------------------------------------------------------
# Building Atari
# ----------------------------------------------------------------------------

add_library(atari_agent STATIC
  src/atari_agent.cpp
  src/atari_env.cpp
  src/utility.cpp
)

# Using PRIVATE in target_compile_options keeps the options local to this library
target_compile_options(atari_agent PRIVATE -Wall -Wextra -Werror -Wno-unused -Wpedantic $<$<CONFIG:RELEASE>:-O2 -flto>)

target_compile_features(atari_agent PRIVATE cxx_std_17)

target_include_directories(atari_agent
  PUBLIC
    $<INSTALL_INTERFACE:include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  PRIVATE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/atari_agent>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
    $<BUILD_INTERFACE:${ale_SOURCE_DIR}/src>
    $<BUILD_INTERFACE:${ale_BINARY_DIR}/src>
)

target_link_libraries(atari_agent
PUBLIC
  drla::drla
  drla::aux
  ${TORCH_LIBRARIES}
  Threads::Threads
  nlohmann_json::nlohmann_json
PRIVATE
  $<BUILD_INTERFACE:spdlog::spdlog>
  ale-lib
)

# ----------------------------------------------------------------------------
# Installing the Atari agent library
# ----------------------------------------------------------------------------

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

set(atari_agent_export_targets atari_agent ale ale-lib)
if(NOT nlohmann_json_FOUND)
	set(atari_agent_export_targets ${atari_agent_export_targets} nlohmann_json)
endif()

install(
  TARGETS ${atari_agent_export_targets}
  EXPORT atari_agent_Targets
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  INCLUDES DESTINATION include
)

write_basic_package_version_file(
  "atari_agentConfigVersion.cmake"
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY SameMajorVersion
)

configure_package_config_file(
  "${PROJECT_SOURCE_DIR}/cmake/atari_agentConfig.cmake.in"
  "${PROJECT_BINARY_DIR}/atari_agentConfig.cmake"
  INSTALL_DESTINATION
  ${CMAKE_INSTALL_PREFIX}/cmake
)

install(
  EXPORT atari_agent_Targets
  FILE atari_agentTargets.cmake
  NAMESPACE atari_agent::
  DESTINATION ${CMAKE_INSTALL_PREFIX}/cmake
)

install(
  FILES "${PROJECT_BINARY_DIR}/atari_agentConfig.cmake"
        "${PROJECT_BINARY_DIR}/atari_agentConfigVersion.cmake"
  DESTINATION ${CMAKE_INSTALL_PREFIX}/cmake
)

install(
  DIRECTORY include/
  DESTINATION include
)
