cmake_minimum_required(VERSION 3.14)

# ----------------------------------------------------------------------------
# DRLA Atari
# ----------------------------------------------------------------------------

project(drla-atari
	VERSION 0.1.0
	DESCRIPTION "DRLA Atari"
	LANGUAGES CXX
)

# ----------------------------------------------------------------------------
# Add sub directories
# ----------------------------------------------------------------------------

add_subdirectory(atari_agent)
add_subdirectory(third_party)
add_subdirectory(atari_train)
add_subdirectory(atari_run)
add_subdirectory(atari_roms)
