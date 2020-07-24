# Add an "uninstall" target

CONFIGURE_FILE ("${PROJECT_CMAKE_DIR}/internal/cmake_uninstall.cmake.in" "${PROJECT_BINARY_DIR}/cmake_uninstall.cmake" IMMEDIATE @ONLY)

ADD_CUSTOM_TARGET (uninstall "${CMAKE_COMMAND}" -P "${PROJECT_BINARY_DIR}/cmake_uninstall.cmake")
