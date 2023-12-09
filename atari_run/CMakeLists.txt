cmake_minimum_required(VERSION 3.14)

# ----------------------------------------------------------------------------
# Atari run
# ----------------------------------------------------------------------------

project(atari_run
	VERSION 0.1.0
	DESCRIPTION "Run agents trained in atari environments"
	LANGUAGES CXX
)

# ----------------------------------------------------------------------------
# Dependencies
# ----------------------------------------------------------------------------

include(${CMAKE_SOURCE_DIR}/cmake/cxxopts.cmake)
include(${CMAKE_SOURCE_DIR}/cmake/spdlog.cmake)

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)
find_package(Torch REQUIRED)

# ----------------------------------------------------------------------------
# Building Atari run cli
# ----------------------------------------------------------------------------

add_executable(atari_run
	src/main.cpp
	src/runner.cpp
)

# Using PRIVATE in target_compile_options keeps the options local to this library
target_compile_options(atari_run PRIVATE -Wall -Wextra -Werror -Wno-unused $<$<CONFIG:RELEASE>:-O2 -flto>)

target_compile_features(atari_run PRIVATE cxx_std_17)

target_include_directories(atari_run
	PRIVATE
		src
		${GifEncoder_SOURCE_DIR}/egif
)

target_link_libraries(atari_run
PUBLIC
	atari_agent
	${TORCH_LIBRARIES}
	Threads::Threads
	cxxopts
	spdlog
)

# ----------------------------------------------------------------------------
# Installing Atari train cli
# ----------------------------------------------------------------------------

install(
	TARGETS atari_run
	RUNTIME DESTINATION bin
)
