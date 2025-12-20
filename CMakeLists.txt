cmake_minimum_required(VERSION 3.14)
project(MyProject)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Opzioni di compilazione
add_compile_options(-Wall -Wextra)

# Trova Eigen3 (necessario per autodiff)
find_package(Eigen3 REQUIRED)

# Configura autodiff prima di aggiungerlo come sottodirectory
# Questo sostituisce FetchContent perché i file sono già presenti
set(AUTODIFF_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(AUTODIFF_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(AUTODIFF_BUILD_PYTHON OFF CACHE BOOL "" FORCE)

# Aggiunge la cartella locale
add_subdirectory(lib/autodiff)

# Definizione degli eseguibili
add_executable(main_app src/main.cpp)
add_executable(test_runner tests/main.cpp)

# Link alla libreria autodiff (il target 'autodiff' è definito all'interno della sua sottodirectory)
target_link_libraries(main_app PRIVATE autodiff::autodiff)
target_link_libraries(test_runner PRIVATE autodiff::autodiff)

# Include directories per il tuo progetto
target_include_directories(main_app PRIVATE ${CMAKE_SOURCE_DIR}/lib)
target_include_directories(test_runner PRIVATE ${CMAKE_SOURCE_DIR}/lib)
