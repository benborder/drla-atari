include(FetchContent)
FetchContent_Declare(
	ale
	GIT_REPOSITORY https://github.com/mgbellemare/Arcade-Learning-Environment.git
	GIT_TAG        master
)
set(BUILD_PYTHON_LIB OFF)
FetchContent_MakeAvailable(ale)
