if(CMAKE_SIZEOF_VOID_P EQUAL 8)
	set(ASSIMP_ARCHITECTURE "64")
elseif(CMAKE_SIZEOF_VOID_P EQUAL 4)
	set(ASSIMP_ARCHITECTURE "32")
endif(CMAKE_SIZEOF_VOID_P EQUAL 8)


if(WIN32)
	set(ASSIMP_ROOT_DIR CACHE PATH "ASSIMP root directory")
    
	# Find path of each library
	find_path(ASSIMP_INCLUDE_DIR
		NAMES
			assimp/anim.h
		HINTS
			${ASSIMP_ROOT_DIR}/include
	)

	if(MSVC12)
		set(ASSIMP_MSVC_VERSION "vc120")
	elseif(MSVC14)	
		set(ASSIMP_MSVC_VERSION "vc140")
	endif(MSVC12)
	
	if(MSVC12 OR MSVC14)
	
		find_path(ASSIMP_LIBRARY_DIR
			NAMES
				assimp-${ASSIMP_MSVC_VERSION}-mt.lib
			HINTS
				${ASSIMP_ROOT_DIR}/lib${ASSIMP_ARCHITECTURE}
		)
		
		find_library(ASSIMP_LIBRARY_RELEASE				assimp-${ASSIMP_MSVC_VERSION}-mt.lib 			PATHS ${ASSIMP_LIBRARY_DIR})
		find_library(ASSIMP_LIBRARY_DEBUG				assimp-${ASSIMP_MSVC_VERSION}-mtd.lib			PATHS ${ASSIMP_LIBRARY_DIR})
		
		set(ASSIMP_LIBRARY 
			optimized 	${ASSIMP_LIBRARY_RELEASE}
			debug		${ASSIMP_LIBRARY_DEBUG}
		)
		
		set(ASSIMP_LIBRARIES "ASSIMP_LIBRARY_RELEASE" "ASSIMP_LIBRARY_DEBUG")
	
		FUNCTION(ASSIMP_COPY_BINARIES TargetDirectory)
			ADD_CUSTOM_TARGET(AssimpCopyBinaries
				COMMAND ${CMAKE_COMMAND} -E copy ${ASSIMP_ROOT_DIR}/bin${ASSIMP_ARCHITECTURE}/assimp-${ASSIMP_MSVC_VERSION}-mtd.dll 	${TargetDirectory}/Debug/assimp-${ASSIMP_MSVC_VERSION}-mtd.dll
				COMMAND ${CMAKE_COMMAND} -E copy ${ASSIMP_ROOT_DIR}/bin${ASSIMP_ARCHITECTURE}/assimp-${ASSIMP_MSVC_VERSION}-mt.dll 		${TargetDirectory}/Release/assimp-${ASSIMP_MSVC_VERSION}-mt.dll
			COMMENT "Copying Assimp binaries to '${TargetDirectory}'"
			VERBATIM)
		ENDFUNCTION(ASSIMP_COPY_BINARIES)
	
	endif()
	
else(WIN32)
    
	find_path(
	  assimp_INCLUDE_DIRS
	  NAMES assimp/postprocess.h assimp/scene.h assimp/version.h assimp/config.h assimp/cimport.h
	  PATHS ASSIMP_INCLUDE_DIRS /usr/include /usr/local/include
	)
    #message(STATUS "Found ASSIMP: ${assimp_INCLUDE_DIRS}")
    #message(STATUS "ASSIMP_INCLUDE_DIRS: ${ASSIMP_INCLUDE_DIRS}")
	find_library(
	  ASSIMP_LIBRARY
	  NAMES assimp
	  PATHS ASSIMP_LIBRARY_DIRS /usr/lib64 /usr/lib
	)
    #message(STATUS "Found ASSIMP: ${ASSIMP_LIBRARY}")
	if (assimp_INCLUDE_DIRS AND ASSIMP_LIBRARY)
	  SET(assimp_FOUND TRUE)
	ENDIF (assimp_INCLUDE_DIRS AND ASSIMP_LIBRARY)

	if (assimp_FOUND)
	  if (NOT assimp_FIND_QUIETLY)
		message(STATUS "Found ASSIMP: ${ASSIMP_LIBRARY}")
	  endif (NOT assimp_FIND_QUIETLY)
	else (assimp_FOUND)
	  if (assimp_FIND_REQUIRED)
		message(FATAL_ERROR "Could not find ASSIMP")
	  endif (assimp_FIND_REQUIRED)
	endif (assimp_FOUND)
	
endif(WIN32)
