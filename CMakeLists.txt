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

# Install the LMPC walking package
SET(SECOND_ORDER_FILES
  __init__.py
  constraints.py
  cost_function.py
  LIPM_to_whole_body.py
  motion_model.py
  plot_utils.py
  reference_trajectories.py
)

SET(LMPC_NAME "lmpc_walking")
PYTHON_INSTALL_ON_SITE(${LMPC_NAME} __init__.py)

FOREACH(python ${SECOND_ORDER_FILES})
  PYTHON_INSTALL_ON_SITE(${LMPC_NAME}/second_order/ ${python})
ENDFOREACH(python ${SECOND_ORDER_FILES})

# Install the hybrid_biped pkg
ADD_SUBDIRECTORY(python)