

#include "render.h"
#include "scene.h"
#include "app_perf.h"

#ifdef BUILD_OPTIX

#include <optix.h>
#include <optixu/optixu.h>
#include <optixu/optixpp_namespace.h>
using namespace optix;

optix::Context	g_OptixContext;
GeometryGroup	g_OptixMainGroup;
Program			g_OptixMeshIntersectProg;
Program			g_OptixMeshBBoxProg;
std::vector< GeometryGroup >	g_OptixGeom;
std::vector< optix::Material >	g_OptixMats;
int				g_OptixTex;

int OPTIX_PROGRAM = -1;	// FTIZB Shader Program
int OPTIX_VIEW;		
int OPTIX_PROJ;		
int OPTIX_MODEL;		
int OPTIX_LIGHTPOS;	
int OPTIX_LIGHTTARGET;
int OPTIX_CLRAMB;
int OPTIX_CLRDIFF;
int OPTIX_CLRSPEC;

optix::Program CreateProgramOptix ( std::string name, std::string prog_func )
{
	optix::Program program;

	nvprintf  ( "  Loading: %s, %s\n", name.c_str(), prog_func.c_str() );
	try { 
		program = g_OptixContext->createProgramFromPTXFile ( name, prog_func );
	} catch (Exception e) {
		nvprintf  ( "  OPTIX ERROR: %s \n", g_OptixContext->getErrorString( e.getErrorCode() ).c_str() );
		nverror ();		
	}
	return program;
}

Buffer CreateOutputOptix ( RTformat format, unsigned int width, unsigned int height )
{
	Buffer buffer;

	GLuint vbo = 0;
	glGenBuffers (1, &vbo );
	glBindBuffer ( GL_ARRAY_BUFFER, vbo );
	size_t element_size;
    g_OptixContext->checkError( rtuGetSizeForRTformat(format, &element_size));
    glBufferData(GL_ARRAY_BUFFER, element_size * width * height, 0, GL_STREAM_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    buffer = g_OptixContext->createBufferFromGLBO(RT_BUFFER_OUTPUT, vbo);
    buffer->setFormat(format);
    buffer->setSize( width, height );

	return buffer;
}

void CreateMaterialOptix ( Material material, std::string filename, std::string ch_name, std::string ah_name )
{
	std::string ptx_file = filename + ".ptx";
	Program ch_program = CreateProgramOptix ( ptx_file, ch_name );
	Program ah_program = CreateProgramOptix ( ptx_file, ah_name );
	material->setClosestHitProgram ( 0, ch_program );
	material->setAnyHitProgram ( 1, ah_program );
}


void renderAddShaderOptix ( Scene& scene, char* vertname, char* fragname )
{
	OPTIX_PROGRAM = scene.AddShader ( vertname, fragname );
	scene.AddParam ( OPTIX_PROGRAM, UVIEW, "uView" );
	scene.AddParam ( OPTIX_PROGRAM, UPROJ, "uProj" );
	scene.AddParam ( OPTIX_PROGRAM, UMODEL, "uModel" );
	scene.AddParam ( OPTIX_PROGRAM, ULIGHTPOS, "uLightPos" );
	scene.AddParam ( OPTIX_PROGRAM, UCLRAMB, "uClrAmb" );
	scene.AddParam ( OPTIX_PROGRAM, UCLRDIFF, "uClrDiff" );
	scene.AddParam ( OPTIX_PROGRAM, UCLRSPEC, "uClrSpec" );

	scene.AddParam ( OPTIX_PROGRAM, UOVERTEX, "uOverTex" );
	scene.AddParam ( OPTIX_PROGRAM, UOVERSIZE, "uOverSize" );
}

void renderInitializeOptix ( int w, int h )
{
	// Create OptiX context
	g_OptixContext = Context::create();
	g_OptixContext->setEntryPointCount ( 1 );
	g_OptixContext->setRayTypeCount( 2 );
	g_OptixContext->setStackSize( 2400 );

	g_OptixContext["scene_epsilon"]->setFloat( 1.0e-6f );
	g_OptixContext["radiance_ray_type"]->setUint( 0u );
	g_OptixContext["shadow_ray_type"]->setUint( 1u );
	g_OptixContext["max_depth"]->setInt( 1 );
	g_OptixContext["frame_number"]->setUint( 0u );

	// Create Output buffer
	Variable outbuf = g_OptixContext["output_buffer"];
	Buffer buffer = CreateOutputOptix( RT_FORMAT_UNSIGNED_BYTE4, w, h );
	outbuf->set ( buffer );

	// Camera ray gen and exception program  
	g_OptixContext->setRayGenerationProgram( 0, CreateProgramOptix( "optix_camera_rays.ptx", "pinhole_camera" ) );
	g_OptixContext->setExceptionProgram(     0, CreateProgramOptix( "optix_camera_rays.ptx", "exception" ) );

	// Used by both exception programs
	g_OptixContext["bad_color"]->setFloat( 0.0f, 1.0f, 1.0f );

	// Assign miss program
	g_OptixContext->setMissProgram( 0, CreateProgramOptix( "optix_miss_rays.ptx", "miss" ) );
	g_OptixContext["background_light"]->setFloat( 1.0f, 1.0f, 1.0f );
	g_OptixContext["background_dark"]->setFloat( 0.3f, 0.3f, 0.3f );

	// Align background's up direction with camera's look direction
	float3 bg_up;  bg_up.x=0; bg_up.y=1; bg_up.z=0;
	g_OptixContext["up"]->setFloat( bg_up.x, bg_up.y, bg_up.z );

	// Variance buffers
	Buffer variance_sum_buffer = g_OptixContext->createBuffer( RT_BUFFER_INPUT_OUTPUT | RT_BUFFER_GPU_LOCAL, RT_FORMAT_FLOAT4, w, h );
    memset( variance_sum_buffer->map(), 0, w*h*sizeof(float4) );
	variance_sum_buffer->unmap();
	g_OptixContext["variance_sum_buffer"]->set( variance_sum_buffer );
	Buffer variance_sum2_buffer = g_OptixContext->createBuffer( RT_BUFFER_INPUT_OUTPUT | RT_BUFFER_GPU_LOCAL, RT_FORMAT_FLOAT4, w, h );
	memset( variance_sum2_buffer->map(), 0, w*h*sizeof(float4) );
	variance_sum2_buffer->unmap();
	g_OptixContext["variance_sum2_buffer"]->set( variance_sum2_buffer );

	// Sample count buffer
	Buffer num_samples_buffer = g_OptixContext->createBuffer( RT_BUFFER_INPUT_OUTPUT | RT_BUFFER_GPU_LOCAL, RT_FORMAT_UNSIGNED_INT, w, h );
    memset( num_samples_buffer->map(), 0, w*h*sizeof(unsigned int) );
	num_samples_buffer->unmap();
	g_OptixContext["num_samples_buffer"]->set( num_samples_buffer);

	// Random seed buffer
	Buffer rnd_seeds = g_OptixContext->createBuffer( RT_BUFFER_INPUT_OUTPUT | RT_BUFFER_GPU_LOCAL, RT_FORMAT_UNSIGNED_INT, w, h );
	unsigned int* seeds = (unsigned int*) rnd_seeds->map();
	for ( int i=0; i < w*h; i++ ) {
		seeds[i] = rand()*0xffffL / RAND_MAX;
	}
	rnd_seeds->unmap();
	g_OptixContext["rnd_seeds"]->set( rnd_seeds );

	// Default rendering flags
	/*g_OptixContext["shadow_enable"]->setUint( m_shadow ? 1 : 0 );	
	g_OptixContext["mirror_enable"]->setUint( m_mirror ? 1 : 0 );
	g_OptixContext["cone_enable"]->setUint( m_cone ? 1 : 0 );*/

	// Initialize main geometry group
	g_OptixMeshIntersectProg =	CreateProgramOptix ( "optix_triangle_mesh.ptx", "mesh_intersect" );
	g_OptixMeshBBoxProg	=		CreateProgramOptix ( "optix_triangle_mesh.ptx", "mesh_bounds" );

	if (g_OptixMeshIntersectProg==0) {
		nvprintf  ( "Error: Unable to load mesh_intersect program.\n" ); 
		nverror ();
	}
	if (g_OptixMeshBBoxProg==0) {
		nvprintf  ( "Error: Unable to load mesh_bounds program.\n" ); 
		nverror ();
	}

	g_OptixMainGroup = g_OptixContext->createGeometryGroup ();
	g_OptixMainGroup->setChildCount ( 0 );
	g_OptixMainGroup->setAcceleration( g_OptixContext->createAcceleration("Bvh","Bvh") );
	g_OptixContext["top_object"]->set( g_OptixMainGroup );

	// Create an output texture for OpenGL
	glGenTextures( 1, (GLuint*) &g_OptixTex );
	glBindTexture( GL_TEXTURE_2D, g_OptixTex );	
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);		// Change these to GL_LINEAR for super- or sub-sampling
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);			
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);	// GL_CLAMP_TO_EDGE for linear filtering, not relevant for nearest.
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glBindTexture( GL_TEXTURE_2D, 0);
}

void renderValidateOptix ()
{
	try {
		g_OptixContext->validate ();
	} catch (const Exception& e) {		
		std::string msg = g_OptixContext->getErrorString ( e.getErrorCode() );		
		nvprintf  ( "OPTIX ERROR:\n%s\n", msg.c_str() );
		nverror ();		
	}
	try {
		g_OptixContext->compile ();
	} catch (const Exception& e) {		
		std::string msg = g_OptixContext->getErrorString ( e.getErrorCode() );		
		nvprintf  ( "OPTIX ERROR:\n%s\n", msg.c_str() );
		nverror ();		
	}
}

int renderAddMaterialOptix ( Scene& scene, std::string name )
{
	// Create Optix material
	optix::Material omat = g_OptixContext->createMaterial();
	int oid = g_OptixMats.size();

	// Add material to scene 
	int mid = scene.AddMaterial ();
	scene.SetMaterialParam ( mid, 0, Vector3DF(oid, 0, 0) );	// Link to optix material id

	CreateMaterialOptix ( omat, name, "closest_hit_radiance", "any_hit_shadow" );

	omat["importance_cutoff"  ]->setFloat( 0.01f );
	omat["cutoff_color"       ]->setFloat( 0.1f, 0.1f, 0.1f );
	omat["reflection_maxdepth"]->setInt( 1 );  
	omat["reflection_color"   ]->setFloat( 0.2f, 0.2f, 0.2f );  
	omat["shadow_attenuation"]->setFloat( 1.0f, 1.0f, 1.0f );
		
	g_OptixMats.push_back ( omat );	

	return oid;
}

void renderAddModelOptix ( Model* model, int oid )
{
	GeometryGroup geom;
	int id = g_OptixGeom.size();

	const char* Builder = "Sbvh";
	const char* Traverser = "Bvh";
	const char* Refine = "0";
	
	int num_vertices = model->vertCount;
	int num_triangles = model->elemCount;
	int num_normals = num_vertices;
	
	geom = g_OptixContext->createGeometryGroup ();

	//------------------ Per-vertex
	// Vertex buffer
	Buffer vbuffer = g_OptixContext->createBuffer( RT_BUFFER_INPUT, RT_FORMAT_FLOAT3, num_vertices );
	float3* vbuffer_data = static_cast<float3*>( vbuffer->map() );

	// Normal buffer
	Buffer nbuffer = g_OptixContext->createBuffer( RT_BUFFER_INPUT, RT_FORMAT_FLOAT3, num_normals );
	float3* nbuffer_data = static_cast<float3*>( nbuffer->map() );

	// Texcoord buffer
    Buffer tbuffer = g_OptixContext->createBuffer( RT_BUFFER_INPUT, RT_FORMAT_FLOAT2, num_vertices );
    float2* tbuffer_data = static_cast<float2*>( tbuffer->map() );

    //------------------ Per-triangle
	// Vertex index buffer
	Buffer vindex_buffer = g_OptixContext->createBuffer ( RT_BUFFER_INPUT, RT_FORMAT_INT3, num_triangles );
	int3* vindex_data = static_cast<int3*>( vindex_buffer->map() );

	// Normal index buffer
	Buffer nindex_buffer = g_OptixContext->createBuffer ( RT_BUFFER_INPUT, RT_FORMAT_INT3, num_triangles );
	int3* nindex_data = static_cast<int3*>( nindex_buffer->map() );

	// Material id buffer 
	 Buffer mindex_buffer = g_OptixContext->createBuffer( RT_BUFFER_INPUT, RT_FORMAT_UNSIGNED_INT, num_triangles );
    unsigned int* mindex_data = static_cast<unsigned int*>( mindex_buffer->map() );

	optix::Geometry mesh;

	// Copy vertex data
	float4 v4;
	float2 tc;
	tc.x = 0; tc.y = 0;
	char* vdat = (char*) model->vertBuffer;
	float3* v3;
	for (unsigned int i=0; i < num_vertices; i++ ) {
		v3 = (float3*) (vdat + model->vertOffset);	
		vbuffer_data[i] = *v3;
		v3 = (float3*) (vdat + model->normOffset);
		nbuffer_data[i] = *v3;
		tbuffer_data[i] = tc;
		vdat += model->vertStride;
	}	

	// Copy element data (indices)
	for (int i=0; i < num_triangles; i++ ) {
		int3 tri_verts;		// vertices in trangle
		tri_verts.x = model->elemBuffer[ i*3   ];
		tri_verts.y = model->elemBuffer[ i*3+1 ];
		tri_verts.z = model->elemBuffer[ i*3+2 ];
		vindex_data [ i ] = tri_verts;
		nindex_data [ i ] = tri_verts;
		mindex_data [ i ] = 0;
	}

	mesh = g_OptixContext->createGeometry ();
	mesh->setPrimitiveCount ( num_triangles );
	mesh->setIntersectionProgram ( g_OptixMeshIntersectProg );
	mesh->setBoundingBoxProgram ( g_OptixMeshBBoxProg );
	
	mesh[ "vertex_buffer" ]->setBuffer( vbuffer );			// num verts
    mesh[ "normal_buffer" ]->setBuffer( nbuffer );	
	mesh[ "texcoord_buffer" ]->setBuffer( tbuffer );	

	mesh[ "vindex_buffer" ]->setBuffer( vindex_buffer );	// num tris
    mesh[ "nindex_buffer" ]->setBuffer( nindex_buffer );
	mesh[ "tindex_buffer" ]->setBuffer( nindex_buffer );
    mesh[ "mindex_buffer" ]->setBuffer( mindex_buffer );

	// Unmap buffers
	vbuffer->unmap();	
	nbuffer->unmap();
	tbuffer->unmap();
	vindex_buffer->unmap();
	nindex_buffer->unmap();
	//tindex_buffer->unmap();
	mindex_buffer->unmap();

	// Create geometry instance
	Material mat;
	mat = g_OptixMats[ oid ];
	GeometryInstance instance = g_OptixContext->createGeometryInstance ( mesh, &mat, &mat+1 );
	//loadMaterialParams ( instance );	

	// Setup geometry group
	int base_child = geom->getChildCount();
	geom->setChildCount ( 1 );
	optix::Acceleration acceleration = g_OptixContext->createAcceleration( Builder, Traverser );
	acceleration->setProperty( "refine", Refine );
	if ( Builder   == std::string("Sbvh") || Builder == std::string("TriangleKdTree") || Traverser == std::string( "KdTree" )) {
      acceleration->setProperty( "vertex_buffer_name", "vertex_buffer" );
      acceleration->setProperty( "index_buffer_name", "vindex_buffer" );
	}
	geom->setAcceleration( acceleration );
	acceleration->markDirty();
	geom->setChild( 0, instance );

	// Add to geometry list
	g_OptixMainGroup->setChildCount ( g_OptixGeom.size()+1 );
	g_OptixMainGroup->setChild ( id, geom->getChild(0) );	
	g_OptixGeom.push_back ( geom );
}

Buffer getOptixBuffer ()
{
	return g_OptixContext["output_buffer"]->getBuffer();
}

#define RAD2DEG  57.2957795131
#define DEG2RAD  0.01745329251

float renderOptix ( Scene& scene, int frame, int spp )
{
	PERF_PUSH ( "Render shadows" );

	// Set camera params for Optix
	Camera3D* cam = scene.getCamera();
	float a = cam->getAspect();
	float fovx = cam->getFov();	
	float fovy =  float(RAD2DEG)*2.0f*atanf( tanf(cam->getFov()*float(DEG2RAD)/2.0f) / a );
	Vector3DF U = cam->getU() * (2.0f*tanf ( fovy*float(DEG2RAD/2.0f) ) * a);
	Vector3DF V = cam->getV() * (2.0f*tanf ( fovy*float(DEG2RAD/2.0f) ) );		 

	g_OptixContext["eye"]->setFloat ( cam->getPos().x, cam->getPos().y, cam->getPos().z );
	g_OptixContext["U"]->setFloat ( U.x, U.y, U.z );
	g_OptixContext["V"]->setFloat ( V.x, V.y, V.z );
	g_OptixContext["W"]->setFloat ( -cam->getW().x, -cam->getW().y, -cam->getW().z );
	g_OptixContext["frame_number"]->setUint ( frame );

	// Set light params for Optix
	Light* light = scene.getLight();
	g_OptixContext["light_pos"]->setFloat ( light->getPos().x, light->getPos().y, light->getPos().z );

	// Set num samples
	g_OptixContext["num_samples"]->setUint ( spp );

	// Get buffer size
	Buffer buffer = getOptixBuffer();
	RTsize bw, bh;	
	buffer->getSize ( bw, bh );	

	// Launch Optix render
	// entry point 0 - pinhole camera
	g_OptixContext->launch ( 0, (int) bw, (int) bh );

	// Transfer output to OpenGL texture
	glBindTexture( GL_TEXTURE_2D, g_OptixTex );

	int vboid = buffer->getGLBOId ();
    glBindBuffer ( GL_PIXEL_UNPACK_BUFFER, vboid );		// Bind to the optix buffer

	RTsize elementSize = buffer->getElementSize();
	if      ((elementSize % 8) == 0) glPixelStorei(GL_UNPACK_ALIGNMENT, 8);
	else if ((elementSize % 4) == 0) glPixelStorei(GL_UNPACK_ALIGNMENT, 4);
	else if ((elementSize % 2) == 0) glPixelStorei(GL_UNPACK_ALIGNMENT, 2);
	else                             glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

	// Copy the OptiX results into a GL texture
	//  (device-to-device transfer using bound gpu buffer)
	RTformat buffer_format = buffer->getFormat();
	switch (buffer_format) {
	case RT_FORMAT_UNSIGNED_BYTE4:	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8,			bw, bh, 0, GL_BGRA, GL_UNSIGNED_BYTE, 0 );	break;
	case RT_FORMAT_FLOAT4:			glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F_ARB,		bw, bh, 0, GL_RGBA, GL_FLOAT, 0);	break;
	case RT_FORMAT_FLOAT3:			glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB32F_ARB,		bw, bh, 0, GL_RGB, GL_FLOAT, 0);		break;
	case RT_FORMAT_FLOAT:			glTexImage2D(GL_TEXTURE_2D, 0, GL_LUMINANCE32F_ARB, bw, bh, 0, GL_LUMINANCE, GL_FLOAT, 0);	break;	
	}
	glBindBuffer( GL_PIXEL_UNPACK_BUFFER, 0 );	
	glBindTexture( GL_TEXTURE_2D, 0);

	//-- Debugging: Pass in a known buffer
	/*char* pix = (char*) malloc ( bw*bh*4 );
	for (int n=0; n < bw*bh*4; n+=4 ) {
		pix[n+0] = rand()*255/RAND_MAX;		// b
		pix[n+1] = 0;		// g
		pix[n+2] = 0;		// r
		pix[n+3] = 255;		// a
	}
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8,			bw, bh, 0, GL_BGRA, GL_UNSIGNED_BYTE, pix );	
	free ( pix );   */

	return PERF_POP ();
}

void renderSetupOptixGL ( Scene& scene, int prog )
{
	// OpenGL specific code to bind the 
	// optix GL texture to the GLSL shader

	Buffer buffer = getOptixBuffer();
	RTsize bw, bh;	
	buffer->getSize ( bw, bh );	
	int sz[2] = { bw, bh };
	glProgramUniform2iv( prog, scene.getParam(prog, UOVERSIZE), 1, sz );     // Set value for "renderSize" uniform in the shader
	
	glProgramUniform1i ( prog, scene.getParam(prog, UOVERTEX), 0 );
	glActiveTexture ( GL_TEXTURE0 );
	glBindTexture ( GL_TEXTURE_2D, g_OptixTex );
}

#endif