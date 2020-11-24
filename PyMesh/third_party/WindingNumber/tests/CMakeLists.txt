SET(SRC_FILES test_main.cpp test_UT_SolidAngle.cpp)
SET(INC_FILES catch.hpp)

ADD_EXECUTABLE(test_fast_winding_number ${SRC_FILES} ${INC_FILES})
TARGET_LINK_LIBRARIES(test_fast_winding_number PRIVATE FastWindingNumber)
TARGET_INCLUDE_DIRECTORIES(test_fast_winding_number SYSTEM PRIVATE ${PROJECT_SOURCE_DIR})

ADD_CUSTOM_TARGET(tests
    DEPENDS test_fast_winding_number
    COMMAND test_fast_winding_number
    )
