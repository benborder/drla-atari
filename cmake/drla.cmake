find_package(drla QUIET)
if(${drla_FOUND})
	message(STATUS "Found drla ${drla_DIR}")
else()
	include(FetchContent)
	FetchContent_Declare(
		drla
		GIT_REPOSITORY https://github.com/benborder/drla.git
		GIT_TAG        master
	)
	FetchContent_MakeAvailable(drla)
endif()
