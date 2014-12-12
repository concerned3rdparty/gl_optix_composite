
#include "model.h"

#include "loader_OBJReader.h"

Model::Model() :
	vertBuffer(0), elemBuffer(0),
	vertArrayID(-1), vertBufferID(-1), elemBufferID(-1)
{
}

Model::~Model ()
{
}


void Model::Transform ( Vector3DF move, Vector3DF scale )
{
	float* pos = (float*) ( (char*) vertBuffer + vertOffset);	

	for (int n=0; n < vertCount; n++ ) {
		*(pos)	 = *(pos) * scale.x + move.x;
		*(pos+1) = *(pos+1) * scale.y + move.y;
		*(pos+2) = *(pos+2) * scale.z + move.z;
		pos += vertStride/sizeof(float);
	}		
	UpdateVBO();
}

void Model::UniqueNormals ()
{
	// Build unique triangles to get flat normals
	int v1, v2, v3;	
	Vector3DF a, b, c, n;

	nvprintf  ( "Generating unique normals..\n" );

	// Create new vert/normal buffers
	Vector3DF* vert_buf = (Vector3DF*) malloc ( (9*elemCount) * 2*3*sizeof(float) );	
	Vector3DF* vert_dest = vert_buf;		

	unsigned int* indx_buf = (unsigned int*) malloc ( elemCount * 3*sizeof(unsigned int) );
	unsigned int* indx_dest = indx_buf;
	Vector3DF* vert_src = (Vector3DF*) (vertBuffer + vertOffset);
	int vm = vertStride / sizeof(Vector3DF);		// stride as multiple of Vec3F

	for (int n=0; n < elemCount; n++ ) {

		// Get vertices 
		v1 = elemBuffer[ n*3 ]; v2 = elemBuffer[ n*3+1 ]; v2 = elemBuffer[ n*3+2 ];		

		// Compute face normal
		a = vert_src[v1*vm]; b = vert_src[v2*vm];	c = vert_src[v3*vm];		
		a -= c;	b -= c; a.Cross ( b );
		a.Normalize ();
		
		// Output vertices and normals
		*vert_dest++ = vert_src[v1*vm];		*vert_dest++ = a;
		*vert_dest++ = vert_src[v2*vm];		*vert_dest++ = a;
		*vert_dest++ = vert_src[v3*vm];		*vert_dest++ = a;		

		// Output new indices
		*indx_dest++ = v1;
		*indx_dest++ = v2;
		*indx_dest++ = v3;
	}

	// Update model data
	free ( vertBuffer );
	free ( elemBuffer );
	
	vertBuffer = (float*) vert_buf;	
	vertStride = 2*3*sizeof(float);
	vertCount = elemCount*3;
	vertComponents = 6;

	elemBuffer = indx_buf;
	elemStride = 3*sizeof(unsigned int);
	
	UpdateVBO ();
}

void Model::UpdateVBO ()
{
	// Create VAO
	if ( vertArrayID == -1 )  glGenVertexArrays ( 1, (GLuint*) &vertArrayID );
	glBindVertexArray ( vertArrayID );


	// Update Vertex VBO
	if ( vertBufferID == -1 ) glGenBuffers( 1, (GLuint*) &vertBufferID );	
	//glBindBuffer ( GL_ARRAY_BUFFER, vertBufferID );	
	//glBufferData ( GL_ARRAY_BUFFER, vertCount * vertStride, vertBuffer, GL_STATIC_DRAW );
	glNamedBufferDataEXT( vertBufferID, vertCount * vertStride, vertBuffer, GL_STATIC_DRAW );
	glEnableVertexAttribArray ( 0 );
	glBindVertexBuffer ( 0, vertBufferID, 0, vertStride );
	glVertexAttribFormat ( 0, vertComponents, GL_FLOAT, false, vertOffset );
	glVertexAttribBinding ( 0, 0 );
	glEnableVertexAttribArray ( 1 );
	glVertexAttribFormat ( 1, normComponents, GL_FLOAT, false, normOffset );
	glVertexAttribBinding ( 1, 0 );
	
	// Update Element VBO
	if ( elemBufferID == -1 ) glGenBuffers( 1, (GLuint*) &elemBufferID );
	//glBindBuffer ( GL_ELEMENT_ARRAY_BUFFER, elemBufferID );
	//glBufferData( GL_ELEMENT_ARRAY_BUFFER, elemCount * elemStride, elemBuffer, GL_STATIC_DRAW );	
	glNamedBufferDataEXT( elemBufferID, elemCount * elemStride, elemBuffer, GL_STATIC_DRAW );	

	glBindVertexArray ( 0 );
	
	glBindBuffer ( GL_ARRAY_BUFFER, 0 );
	glBindBuffer ( GL_ELEMENT_ARRAY_BUFFER, 0 );

}