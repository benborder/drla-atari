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
# Fetching 3rd party libs
# ----------------------------------------------------------------------------

include(FetchContent)

FetchContent_Declare(
	cxxopts
	GIT_REPOSITORY https://github.com/jarro2783/cxxopts.git
	GIT_TAG        v3.0.0
)
FetchContent_MakeAvailable(cxxopts)

FetchContent_Declare(
	spdlog
	GIT_REPOSITORY https://github.com/gabime/spdlog.git
	GIT_TAG        v1.10.0
)
FetchContent_MakeAvailable(spdlog)

FetchContent_Declare(
	fmt
	GIT_REPOSITORY https://github.com/fmtlib/fmt.git
	GIT_TAG        8.1.1
)
FetchContent_MakeAvailable(fmt)

FetchContent_Declare(
	GifEncoder
	GIT_REPOSITORY https://github.com/xiaozhuai/GifEncoder.git
	GIT_TAG        master
)
FetchContent_MakeAvailable(GifEncoder)

# ----------------------------------------------------------------------------
# Building Atari run cli
# ----------------------------------------------------------------------------

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)
find_package(Torch REQUIRED)

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
		${fmt_SOURCE_DIR}
		${GifEncoder_SOURCE_DIR}/egif
)

target_link_libraries(atari_run
PUBLIC
	atari_agent
	${TORCH_LIBRARIES}
	Threads::Threads
	cxxopts
	egif
	spdlog
	fmt
)

# ----------------------------------------------------------------------------
# Installing Atari train cli
# ----------------------------------------------------------------------------

install(
	TARGETS atari_run
	RUNTIME DESTINATION bin
)
