cmake_minimum_required(VERSION 3.20)
project(SkMetalBridge LANGUAGES Swift C)

# Set Swift standard
set(CMAKE_Swift_STANDARD 6)
set(CMAKE_Swift_STANDARD_REQUIRED ON)

# Find Swift compiler
find_program(SWIFT_COMPILER swift)
if(NOT SWIFT_COMPILER)
    message(FATAL_ERROR "Swift compiler not found. Please install Xcode or Swift toolchain.")
endif()

# Get Swift version
execute_process(
    COMMAND ${SWIFT_COMPILER} --version
    OUTPUT_VARIABLE SWIFT_VERSION_OUTPUT
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
message(STATUS "Swift version: ${SWIFT_VERSION_OUTPUT}")

# Build Swift package using swift build
add_custom_target(skmetal_bridge_build
    COMMAND swift build --configuration release --package-path ${CMAKE_CURRENT_SOURCE_DIR}
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMENT "Building SkMetalBridge Swift package"
    VERBATIM
)

# Locate the built dylib
set(SKMETAL_DYLIB_PATH "${CMAKE_CURRENT_SOURCE_DIR}/.build/release/libSkMetalBridge.dylib")
set(SKMETAL_C_DYLIB_PATH "${CMAKE_CURRENT_SOURCE_DIR}/.build/release/libSkMetalBridgeC.dylib")

# Copy dylibs to Python package location
add_custom_command(
    TARGET skmetal_bridge_build POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${SKMETAL_DYLIB_PATH}
        ${CMAKE_CURRENT_SOURCE_DIR}/../skmetal/libSkMetalBridge.dylib
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${SKMETAL_C_DYLIB_PATH}
        ${CMAKE_CURRENT_SOURCE_DIR}/../skmetal/libSkMetalBridgeC.dylib
    COMMENT "Copying dylibs to Python package"
)

# Install target for the Python package
add_custom_target(install_skmetal_bridge
    DEPENDS skmetal_bridge_build
    COMMENT "SkMetalBridge ready for Python package"
)

# Make the install target depend on our custom target
add_dependencies(install install_skmetal_bridge)