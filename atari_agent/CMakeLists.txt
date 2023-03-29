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
# Fetching 3rd party libs
# ----------------------------------------------------------------------------

include(FetchContent)

FetchContent_Declare(
  nlohmann_json
  GIT_REPOSITORY https://github.com/nlohmann/json.git
  GIT_TAG        v3.10.5
)
FetchContent_MakeAvailable(nlohmann_json)

FetchContent_Declare(
  spdlog
  GIT_REPOSITORY https://github.com/gabime/spdlog.git
  GIT_TAG        v1.10.0
)
FetchContent_MakeAvailable(spdlog)

FetchContent_Declare(
	drla
	GIT_REPOSITORY https://github.com/benborder/drla.git
	GIT_TAG        master
)
FetchContent_MakeAvailable(drla)

FetchContent_Declare(
	ale
	GIT_REPOSITORY https://github.com/mgbellemare/Arcade-Learning-Environment.git
	GIT_TAG        master
)
set(BUILD_PYTHON_LIB OFF)
FetchContent_MakeAvailable(ale)

# ----------------------------------------------------------------------------
# Building Atari
# ----------------------------------------------------------------------------

set(THREADS_PREFER_PTHREAD_FLAG ON)  # Set -pthreads
find_package(Threads REQUIRED)
find_package(Torch REQUIRED)

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
