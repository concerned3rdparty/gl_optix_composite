/**************************************************************
** ObjarReader.cpp                                           **
** ---------------------                                     **
**                                                           **
** This file loads a .objar file, which is a binary model    **
**     format used by Chris Wyman's IGLU library.  Since     **
**     we cannot release the entire library, this code does  **
**     an extremely simplisitc load of this format into the  **
**     sample NVIZBModelInput format plus a pair of OpenGL   **
**     buffer objects.                                       **
** This code assumes v2 of the .objar format (2014), which   **
**     shouldn't be a problem, unless you grab the IGLU code **
**     posted on his University of Iowa website to create    **
**     models with the older format.                         **
**                                                           **
** Chris Wyman (9/2/2014)                                    **
**************************************************************/

#ifndef OBJAR_READER
	#define OBJAR_READER

	#include <stdio.h>
	#include <stdlib.h>
	#include <GL/glew.h>

	#include "model.h"

	// Header for the IGLU library's .objar binary object file
	//    -> Not real important for the purposes of this sample (other than we have to read this to read the model)
	typedef struct {
			unsigned int vboVersionID;       // Magic number / file version header
			unsigned int numVerts;           // Number of vertices in the VBO
			unsigned int numElems;           // Number of elements (i.e., indices in the VBO)
			unsigned int elemType;           // E.g. GL_TRIANGLES
			unsigned int vertStride;         // Number of bytes between subsequent vertices
			unsigned int vertBitfield;       // Binary bitfield describing valid vertex components 
			unsigned int matlOffset;         // In vertex, offset to material ID
			unsigned int objOffset;          // In vertex, offset to object ID
			unsigned int vertOffset;         // In vertex, offset to vertex 
			unsigned int normOffset;         // In vertex, offset to normal 
			unsigned int texOffset;          // In vertex, offset to texture coordinate
			char matlFileName[84];           // Filename containing material information
			float bboxMin[3];                // Minimum point on an axis-aligned bounding box
			float bboxMax[3];                // Maximum point on an axis-aligned bounding box
			char pad[104];                   // padding to be used later!
	} OBJARHeader;


	class OBJARReader {
	public:
		int LoadHeader( FILE *fp, OBJARHeader *hdr );		
		bool LoadFile ( Model* model, const char *filename, char** searchPaths, int numPaths );
		static bool isMyFile ( const char* filename );		

		friend Model;

	};
#endif

