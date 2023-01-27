CMAKE_MINIMUM_REQUIRED(VERSION 3.1)

### Change the name of the project here ###
SET(PROJECT_NAME hybrid-biped)
SET(PROJECT_ORG glunardi)
SET(PROJECT_DESCRIPTION "Hybrid linear inverted pendulum model for robotic's biped applications")
SET(PROJECT_URL https://github.com/${PROJECT_ORG}/${PROJECT_NAME})

INCLUDE(cmake/base.cmake)
INCLUDE(cmake/test.cmake)
INCLUDE(cmake/python.cmake)

SET(CMAKE_INSTALL_PREFIX "$ENV{HOME}/devel/install" CACHE PATH "Install path prefix")

COMPUTE_PROJECT_ARGS(PROJECT_ARGS LANGUAGES CXX)
PROJECT(${PROJECT_NAME} ${PROJECT_ARGS})

FINDPYTHON()

ADD_SUBDIRECTORY(python)