cmake_minimum_required(VERSION 3.14)

# ----------------------------------------------------------------------------
# Atari roms
# ----------------------------------------------------------------------------

project(atari_roms
	VERSION 0.1.0
	DESCRIPTION "Atari roms"
	LANGUAGES CXX
)

if (ROMS_URL)
	include(FetchContent)

	FetchContent_Declare(
		roms
		URL            ${ROMS_URL}
		URL_HASH       MD5=fc9d49b55a974f63c8f8659a4f7c445c
	)
	FetchContent_Populate(roms)

	# This coppies to build directory to make running in debug mode easier
	add_custom_target(atari_roms ALL
		COMMAND ${CMAKE_COMMAND} -E copy_directory ${roms_SOURCE_DIR} ./atari_train/roms
		COMMAND ${CMAKE_COMMAND} -E copy_directory ${roms_SOURCE_DIR} ./atari_run/roms
		COMMENT "Copying Roms"
		WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
	)
	add_dependencies(atari_train atari_roms)
	add_dependencies(atari_run atari_roms)

	install(
		DIRECTORY ${roms_SOURCE_DIR}/
		DESTINATION ${CMAKE_INSTALL_PREFIX}/bin/roms
		FILES_MATCHING PATTERN "*.bin"
	)

endif()
