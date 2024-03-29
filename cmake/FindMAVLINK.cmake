INCLUDE(FindPackageHandleStandardArgs)
INCLUDE(HandleLibraryTypes)

SET(MAVLINK_IncludeSearchPaths
  ./../mavlink/include/v1.0/pixhawk
  ./../mavlink/include/v1.0
  ./../mavlink/include/v1.0/common
)

FIND_PATH(MAVLINK_INCLUDE_DIR
  NAMES mavlink.h
  PATHS ${MAVLINK_IncludeSearchPaths}
)

FIND_PATH(MAVLINK_TYPES_INCLUDE_DIR
  NAMES mavlink_types.h
  PATHS ${MAVLINK_IncludeSearchPaths}
)

# Handle the REQUIRED argument and set the <UPPERCASED_NAME>_FOUND variable
FIND_PACKAGE_HANDLE_STANDARD_ARGS(MAVLINK "Could NOT find MAVLink protocol (mavlink.h)"
  MAVLINK_INCLUDE_DIR
  MAVLINK_TYPES_INCLUDE_DIR
)

MARK_AS_ADVANCED(
  MAVLINK_INCLUDE_DIR
  MAVLINK_TYPES_INCLUDE_DIR
)
