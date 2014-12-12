

#ifndef DEF_MODEL_H
	#define DEF_MODEL_H

	#include "app_util.h"

	class Model {
	public:
		Model();
		~Model();

		void	Transform ( Vector3DF move, Vector3DF scale );
		void	UniqueNormals ();
		void	UpdateVBO ();

	public:
		
		int		elemDataType;
		int		elemCount;		
		int		elemStride;
		int		elemArrayOffset;
		
		int		vertCount;
		int		vertDataType;
		int		vertComponents;		
		int		vertStride;
		int		vertOffset;

		int		normDataType;
		int		normComponents;
		int		normOffset;

		int		vertArrayID, vertBufferID, elemBufferID;
		
		float*			vertBuffer;
		unsigned int*	elemBuffer;

		Vector4DF		clrAmb;
		Vector4DF		clrDiff;
		Vector4DF		clrSpec;	
	};


#endif