cmake_minimum_required(VERSION 3.5)

project(specsim VERSION 1.1)

# Check for macOS
if(APPLE)
    set(MACOS 1)
else()
    set(MACOS 0)
endif()

# Check for Linux
if(CMAKE_COMPILER_IS_GNUCXX)
    set(LINUX 1)
else()
    set(LINUX 0)
endif()

if(NOT CMAKE_COMPILER_IS_GNUCXX AND NOT MACOS)
  message(FATAL_ERROR "This program works only on Mac or Linux")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(Boost_USE_MULTITHREADED TRUE)
option(WITH_OPENMP "OpenMP Library" ON)
option(BUILD_ON_DALMA "Building for NYU DALMA Supercomputer" OFF)

# Set default build type to Debug if not specified
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug)
endif()

# Configure compiler flags based on build type
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    message(STATUS "Building in Debug mode")
    set(CMAKE_CXX_FLAGS "-g -O0") # Enable debug symbols and disable optimizations
elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
    message(STATUS "Building in Release mode")
    set(CMAKE_CXX_FLAGS "-O2") # Enable optimizations for release
endif()

# Configure the header file
# configure_file(rgen_config.h.in rgen_config.h)

if(NOT BUILD_ON_DALMA)
    if(CMAKE_COMPILER_IS_GNUCXX)
        # Add OpenMP flags if enabled
        if(WITH_OPENMP)
            string(APPEND CMAKE_CXX_FLAGS " -fopenmp")
        endif()
    endif()
    if (APPLE)
        # Add OpenMP flags for macOS if enabled
        if(WITH_OPENMP)
            string(APPEND CMAKE_CXX_FLAGS " -fopenmp=libomp")
        endif()
    endif()
else()
    message(STATUS "Building for NYU DALMA Supercomputer...")
    message(WARNING "   - Remember to load the modules required for DALMA before running cmake")
    message(STATUS  "   - Adding optimisation flags for DALMA...")
    # SSE / AVX switch
    option(USE_AVX "Build with AVX support" ON)
    if(USE_AVX)
      set(CMAKE_CXX_FLAGS "-fopenmp -O3 -mavx2 -mfma -march=haswell")
    else()
      set(CMAKE_CXX_FLAGS "-O3 -msse4.2 -march=westmere")
    endif()
endif()

# Bring the core program sources into the project
set(SOURCES_MAIN
    iterative_artificial_spectrum.cpp
    artificial_spectrum.cpp
    models_database.cpp
    derivatives_handler.cpp
    interpol.cpp
    linfit.cpp
    linspace.cpp
    plots_diags.cpp
    build_lorentzian.cpp
    function_rot.cpp
    io_star_params.cpp
    io_star_common.cpp
    noise_models.cpp
    ioproc.cpp
    random_JB.cpp
    combi.cpp
    stellar_models.cpp
    models_database_grid.cpp
    format.cpp
    acoefs.cpp
    external/Alm/Alm_cpp/activity.cpp
    external/ARMM/bump_DP.cpp
    external/ARMM/solver_mm.cpp
    external/rescale/rescale_freqs.cpp
    external/rescale/decompose_nu.cpp
    external/rescale/data.h
    )

set(SOURCES_TEST_LORENTZIAN
    test_fct/lorentzian_tests.cpp
    function_rot.cpp
    ioproc.cpp
    build_lorentzian.cpp
    linspace.cpp
    acoefs.cpp
    external/Alm/Alm_cpp/activity.cpp
    )

# Check and Load dependencies
find_package(Gnuplot REQUIRED)
if(NOT GNUPLOT_FOUND)
    message(FATAL_ERROR "Gnuplot is required for rendering. Cannot pursue. Please install gnuplot")
endif()

if (WITH_OPENMP)
    find_package(OpenMP)
    if (OPENMP_FOUND)
        if(WITH_OPENMP)
            string(APPEND CMAKE_CXX_FLAGS_RELEASE " -fopenmp")
        endif()
    endif()
endif()

# Dependencies that are not portable (BOOST)
set(Boost_USE_MULTITHREADED=OFF)
find_package(Boost COMPONENTS system filesystem iostreams program_options REQUIRED)
include_directories(${Boost_INCLUDE_DIRS})
link_directories(${Boost_LIBRARY_DIRS})

add_executable(${CMAKE_PROJECT_NAME}  ${SOURCES_MAIN})
if(MACOS)
  find_package(Eigen3 REQUIRED NO_MODULE)
  target_compile_definitions(${CMAKE_PROJECT_NAME}  PRIVATE MACOS)
  target_link_libraries(${CMAKE_PROJECT_NAME}  Eigen3::Eigen  ${Boost_LIBRARIES} "-framework CoreFoundation")
elseif(LINUX)
  # Look for eigen and explicitly specify to use it. EIGEN3_INCLUDE_DIR Must be specified in the bashrc
  include_directories( "$ENV{EIGEN3_INCLUDE_DIR}" )
  target_compile_definitions(${CMAKE_PROJECT_NAME}  PRIVATE LINUX)
  target_link_libraries(${CMAKE_PROJECT_NAME}  ${Boost_LIBRARIES}  rt)
endif()