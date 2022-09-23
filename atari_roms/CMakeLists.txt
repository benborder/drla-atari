cmake_minimum_required(VERSION 3.14)

# ----------------------------------------------------------------------------
# Atari roms
# ----------------------------------------------------------------------------

project(atari_roms
	VERSION 0.1.0
	DESCRIPTION "Atari roms"
	LANGUAGES CXX
)

include(FetchContent)

if (ROMS_URL)

	FetchContent_Declare(
		roms
		URL            ${ROMS_URL}
		URL_HASH       MD5=1ef434fb85c6f2700e1d69787ef6f00c
		DOWNLOAD_NO_EXTRACT true
	)
	FetchContent_Populate(roms)

	add_custom_target(atari_roms ALL
		COMMAND 7z x -y ${roms_SOURCE_DIR}/Roms.rar > /dev/null
		COMMENT "Unpacking ROMS"
		WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/atari_roms
	)

	add_custom_command(
		TARGET atari_roms
		COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_BINARY_DIR}/atari_roms/ROMS ./atari_train/ROMS
		COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_BINARY_DIR}/atari_roms/ROMS ./atari_run/ROMS
		COMMENT "Copying Roms"
		WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
	)
	add_dependencies(atari_train atari_roms)
	add_dependencies(atari_run atari_roms)

	install(
		DIRECTORY ${CMAKE_BINARY_DIR}/atari_roms/ROMS
		DESTINATION ${CMAKE_INSTALL_PREFIX}/bin
		FILES_MATCHING PATTERN "*.bin"
	)

endif()
