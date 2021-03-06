Project(NEAT)

ADD_DEFINITIONS(
    -DBUILD_NEAT_DLL 
    -DTIXML_USE_STL
	-DNOMINMAX
	-DBOOST_ALL_NO_LIB
	-DBOOST_PYTHON_STATIC_LIB
	-D_USE_MATH_DEFINES
)

if(CYGWIN)
	ADD_DEFINITIONS(
		-DWIN32
		-D_CONSOLE
		-D__WXMSW__
		-DWXUSINGDLL
		-DNOPCH
		)
ELSE(CYGWIN)
	if(WIN32)
		ADD_DEFINITIONS(
			-DWIN32
			-D_CONSOLE
			-D__WXMSW__
			-DWXUSINGDLL
			-DNOPCH
			)
	else(WIN32)
		IF(APPLE)
		ADD_DEFINITIONS(
			-D__WXMAC__
			-DWXUSINGDLL
			-DNOPCH
			)
		ELSE(APPLE)
		ADD_DEFINITIONS(
            -D__WXGTK__
            -D__WXGTK20__
			-DWXUSINGDLL
			-DNOPCH
			)
		ENDIF(APPLE)
	endif(WIN32)
endif(CYGWIN)


INCLUDE_REGULAR_EXPRESSION( "^.*[Nn][Ee][Aa][Tt][_].*$" )

#GET_CMAKE_PROPERTY(CMAKETESTPROP CACHE_VARIABLES)

SET(
TINYXMLDLL_INCLUDE
"../../../tinyxmldll/include"
CACHE
PATH 
"TinyXMLDLL include directory"
) 

SET(
BOOST_ROOT
"../../../../Libraries/boost-trunk"
CACHE
PATH 
"Boost Base Directory"
)

SET(
JGTL_INCLUDE
"../../../JGTL/include"
CACHE
PATH 
"JGTL include directory"
) 

SET(
ZLIB_INCLUDE
"../../../zlib"
CACHE
PATH 
"Zlib include directory"
)

SET(
    NEAT_DEFORMS_PATH
    "${CMAKE_SOURCE_DIR}/../../../../src/hyperneat"
    CACHE
    PATH
    "Directory of C++ additions in Neat-Deforms project."
    FORCE
)

INCLUDE_DIRECTORIES(
    include
    ../NEAT/include
    ../Hypercube_NEAT/include
	${TINYXMLDLL_INCLUDE}
	${BOOST_ROOT}
	${WXWIDGETS_INCLUDE}
	${WXWIDGETS_BUILD_INCLUDE_DEBUG}
	${WXWIDGETS_BUILD_INCLUDE_RELEASE}
	${JGTL_INCLUDE}
	${MPI_INCLUDE_PATH}
	${MPI_INCLUDE}
	${ZLIB_INCLUDE}
	${PYTHON_INCLUDE}
	${NEAT_DEFORMS_PATH}
)

SET(
	BOOST_STAGE
	"${CMAKE_CURRENT_SOURCE_DIR}/../../../../Libraries/boost-trunk/stage/lib"
	CACHE
	PATH 
	"Boost Stage Directory"
)

SET(
TINYXMLDLL_LIB
"${CMAKE_CURRENT_SOURCE_DIR}/../../../tinyxmldll/out/"
CACHE
PATH 
"TinyXMLDLL Library Directory"
) 

SET(
ZLIB_LIB
"${CMAKE_CURRENT_SOURCE_DIR}/../../../zlib/out/"
CACHE
PATH 
"Zlib Library Directory"
) 

LINK_DIRECTORIES(
	${TINYXMLDLL_LIB}
	${ZLIB_LIB}
	${BOOST_STAGE}
	${WXWIDGETS_DEBUG_LIB}
	${WXWIDGETS_RELEASE_LIB}
	${MPI_LIB}
	${PYTHON_LIB}
	${LIBRARY_OUTPUT_PATH}
	${BOARD_LIB}
	${OPENCL_LIB}
	${NEAT_DEFORMS_PATH}
)

ADD_LIBRARY(
PyHyperNEAT
SHARED 

src/PyHyperNEAT.cpp
${NEAT_DEFORMS_PATH}/ImageExperiment/ImageExperiment.cpp
${NEAT_DEFORMS_PATH}/ImageExperiment/ImageExperiment.h
)

SET_TARGET_PROPERTIES(PyHyperNEAT PROPERTIES DEBUG_POSTFIX _d)

SET_TARGET_PROPERTIES(PyHyperNEAT PROPERTIES DEBUG_POSTFIX _d)

IF(WIN32)
    SET_TARGET_PROPERTIES(PyHyperNEAT PROPERTIES SUFFIX .pyd)
ELSE(WIN32)
IF(APPLE)
    SET_TARGET_PROPERTIES(PyHyperNEAT PROPERTIES SUFFIX .so)
ENDIF(APPLE)
    SET_TARGET_PROPERTIES(PyHyperNEAT PROPERTIES PREFIX "")
ENDIF(WIN32)

TARGET_LINK_LIBRARIES(
PyHyperNEAT

debug
Hypercube_NEAT_Base_d

optimized
Hypercube_NEAT_Base

	debug
	NEATLib_d

	optimized
	NEATLib
	
	debug
	tinyxmlpluslib_d

	optimized
	tinyxmlpluslib

	debug
	zlib_d
	
	optimized
	zlib

	debug
	board_d

	optimized
	board

	debug
	${BOOST_LIB_PREFIX_NAME}boost_thread

	optimized
	${BOOST_LIB_PREFIX_NAME}boost_thread

    debug
    ${BOOST_LIB_PREFIX_NAME}boost_python

    optimized
    ${BOOST_LIB_PREFIX_NAME}boost_python

    debug
    ${BOOST_LIB_PREFIX_NAME}boost_filesystem

    optimized
    ${BOOST_LIB_PREFIX_NAME}boost_filesystem

    debug
    ${BOOST_LIB_PREFIX_NAME}boost_system

    optimized
    ${BOOST_LIB_PREFIX_NAME}boost_system
)

IF(USE_GUI)
	IF(CYGWIN)
		TARGET_LINK_LIBRARIES(
			PyHyperNEAT

			debug
			wx_gtk2-2.8

			optimized
			wx_gtk2-2.8
			)
	ELSE(CYGWIN)
		IF(WIN32)
			TARGET_LINK_LIBRARIES(
				PyHyperNEAT 

				debug
				wxbase28d

				optimized
				wxbase28

				debug
				wxmsw28d_core

				optimized
				wxmsw28_core

				debug
				wxmsw28d_adv

				optimized
				wxmsw28_adv
				)
		ELSE(WIN32)
			IF(APPLE)
			TARGET_LINK_LIBRARIES(
				PyHyperNEAT 

				general
				"-framework Carbon"
		
				debug
				wx_gtk2-2.8

				optimized
				wx_gtk2-2.8
				)
			ELSE(APPLE)
			TARGET_LINK_LIBRARIES(
				PyHyperNEAT 

				debug
				wx_gtk2-2.8

				optimized
				wx_gtk2-2.8
				)
			ENDIF(APPLE)
		ENDIF(WIN32)
	ENDIF(CYGWIN)
ENDIF(USE_GUI)

IF(WIN32)
TARGET_LINK_LIBRARIES(
	PyHyperNEAT
	
	general
	python27
	)
ELSE(WIN32)
	IF(APPLE)
	TARGET_LINK_LIBRARIES(
		PyHyperNEAT
	
		general
		"-framework Python"
		)
	ELSE(APPLE)
	TARGET_LINK_LIBRARIES(
		PyHyperNEAT
	
		general
		python2.7
		)
	ENDIF(APPLE)
ENDIF(WIN32)

IF(BUILD_GPU)
	IF(APPLE)
	TARGET_LINK_LIBRARIES(
		PyHyperNEAT
	
		general
		"-framework OpenCL"
	)
	ELSE(APPLE)
	TARGET_LINK_LIBRARIES(
		PyHyperNEAT
	
		general
		OpenCL
	)
	ENDIF(APPLE)
ENDIF(BUILD_GPU)

ADD_DEPENDENCIES(PyHyperNEAT NEATLib Hypercube_NEAT_Base)

