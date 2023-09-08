cmake_minimum_required(VERSION 3.14)

# ----------------------------------------------------------------------------
# Atari train
# ----------------------------------------------------------------------------

project(atari_train
	VERSION 0.1.0
	DESCRIPTION "Atari training application"
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
# Building Atari train cli
# ----------------------------------------------------------------------------

add_executable(atari_train
	src/main.cpp
	src/logger.cpp
)

# Using PRIVATE in target_compile_options keeps the options local to this library
target_compile_options(atari_train PRIVATE -Wall -Wextra -Werror -Wno-unused $<$<CONFIG:RELEASE>:-O2 -flto>)

target_compile_features(atari_train PRIVATE cxx_std_17)

target_include_directories(atari_train
	PRIVATE
		src
)

target_link_libraries(atari_train
PUBLIC
	atari_agent
	${TORCH_LIBRARIES}
	Threads::Threads
	cxxopts
	spdlog
)

# ----------------------------------------------------------------------------
# Installing atari train cli
# ----------------------------------------------------------------------------

install(
	TARGETS atari_train
	RUNTIME DESTINATION bin
)
