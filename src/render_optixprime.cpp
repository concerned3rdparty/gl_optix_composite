

#include "render.h"
#include "scene.h"
#include "app_perf.h"

#define OP_SPP		

#ifdef BUILD_OPTIX_PRIME

#include "render_optixprime.h"
#include <optix_prime/optix_prime.h>
#include <cuda_runtime.h>

RTPcontext		g_OPContext;
std::vector<RTPmodel> g_OPModels;
RTPbufferdesc	g_OPRaysDesc[2];
RTPbufferdesc	g_OPHitsDesc[2];
Buffer<Ray>		g_OPRays[2];
Buffer<Hit>		g_OPHits[2];
int				g_OPTex;
unsigned char*  g_OPShadowData;
int				g_OPSamples;

int OPTIXPRIME_PROGRAM = -1;	// FTIZB Shader Program
int OPTIXPRIME_VIEW;		
int OPTIXPRIME_PROJ;		
int OPTIXPRIME_MODEL;		
int OPTIXPRIME_LIGHTPOS;	
int OPTIXPRIME_LIGHTTARGET;
int OPTIXPRIME_CLRAMB;
int OPTIXPRIME_CLRDIFF;
int OPTIXPRIME_CLRSPEC;

void CHK_PRIME( RTPresult result )
{
  if( result != RTP_SUCCESS ) {
	const char* err_string; 
	rtpContextGetLastErrorString( g_OPContext, &err_string );
	nvprintf  ( "OPTIX PRIME ERROR:\n" );
	nvprintf  ( "  %s (%d)", err_string, (int) result );	
	nverror ();
  }                                                                          
}


void renderAddShaderOptixPrime ( Scene& scene, char* vertname, char* fragname )
{
	OPTIXPRIME_PROGRAM = scene.AddShader ( vertname, fragname );
	scene.AddParam ( OPTIXPRIME_PROGRAM, UVIEW, "uView" );
	scene.AddParam ( OPTIXPRIME_PROGRAM, UPROJ, "uProj" );
	scene.AddParam ( OPTIXPRIME_PROGRAM, UMODEL, "uModel" );
	scene.AddParam ( OPTIXPRIME_PROGRAM, ULIGHTPOS, "uLightPos" );
	scene.AddParam ( OPTIXPRIME_PROGRAM, UCLRAMB, "uClrAmb" );
	scene.AddParam ( OPTIXPRIME_PROGRAM, UCLRDIFF, "uClrDiff" );
	scene.AddParam ( OPTIXPRIME_PROGRAM, UCLRSPEC, "uClrSpec" );

	scene.AddParam ( OPTIXPRIME_PROGRAM, UOVERTEX, "uOverTex" );
	scene.AddParam ( OPTIXPRIME_PROGRAM, UOVERSIZE, "uOverSize" );
}

#define RAD2DEG  57.2957795131
#define DEG2RAD  0.01745329251

void CreatePrimaryRays ( Buffer<Ray>& raysBuffer, Scene& scene, int width, int height )
{
	Camera3D* cam = scene.getCamera ();
	Vector3DF pos, dir;
	Vector3DF U, V, W;
	float u, v;	

	float a = cam->getAspect();
	float fovx = cam->getFov();	
	float fovy =  float(RAD2DEG)*2.0f*atanf( tanf(cam->getFov()*float(DEG2RAD)/2.0f) / a );
	U = cam->getU() * (2.0f*tanf ( fovy*float(DEG2RAD/2.0f) ) * a);
	V = cam->getV() * (2.0f*tanf ( fovy*float(DEG2RAD/2.0f) ) );	
	W = cam->getW(); W *= -1.0f;
	pos = cam->getPos ();

	Ray* rays = raysBuffer.ptr();		
	for( int iy=0; iy < height; iy++ ) {
		for( int ix=0; ix < width; ix++ ) {
			// Construct primary ray 
			u = float(ix)/float(width) - 0.5f;
			v = float(iy)/float(height) - 0.5f;
			dir = U * u;
			dir += V * v;
			dir += W;
			dir.Normalize ();
			Ray r = { make_float3(pos.x,pos.y,pos.z), 0, make_float3(dir.x,dir.y,dir.z), 1e34f };
			*rays++ = r;
		}
	}  
}


 

void CreateShadowRays ( Buffer<Ray>& raysBuffer, Scene& scene, int width, int height, Buffer<Ray>& primaryRays, Buffer<Hit>& primaryHits, int spp, int3* modelElems, Vector3DF* modelVerts)
{
	float eps = 1.0e-2;

	Vector3DF pos, dir, rvec, rpos;
	Ray* primRay = primaryRays.ptr();
	Hit* primHit = primaryHits.ptr();
	Ray* second = raysBuffer.ptr();		
	
	Vector3DF light_pos = scene.getLight()->getPos();
	float radius;

	
	for( int iy=0; iy < height; iy++ ) {
		for( int ix=0; ix < width; ix++ ) {

			if ( primHit->t < 0.0f ) {
				
				for (int s=0; s < g_OPSamples; s++ ) {					
					Ray r = { make_float3(0,0,0), 0, make_float3(0,0,0), 0 };
					*second++ = r;
				}

			} else {
				
				// Determine world coordinate
				pos = Vector3DF( primRay->dir.x, primRay->dir.y, primRay->dir.z );
				pos *= primHit->t;
				pos += Vector3DF( primRay->origin.x, primRay->origin.y, primRay->origin.z );
			
				// Get the surface normal
				int3 tri  = modelElems[ primHit->triId] ;
				Vector3DF v0 = modelVerts[ tri.x*2 ];
				Vector3DF v1 = modelVerts[ tri.y*2 ];
				Vector3DF v2 = modelVerts[ tri.z*2 ];
				v1 -= v0;
				v2 -= v0;			
				Vector3DF norm = v1;
				norm.Cross ( v2 );
				norm.Normalize ();
			
				// Construct shadow rays
				for (int s=0; s < g_OPSamples; s++ ) {									
					radius = (0.125 * primHit->t / 1280.0);			// approximation for fragment partial derivs /w respect to pixel	 
					rvec.Random ( -1, 1, -1, 1, -1, 1 );				
					rvec *= radius;
					
					dir = light_pos;								// ray direction, toward light
					dir -= pos;
					dir.Normalize ();				

					rpos = pos;										// ray origin, jittered for subsampling
					rpos += rvec;	
					rpos += norm * float(radius*2.0);

					
					Ray r = { make_float3(rpos.x,rpos.y,rpos.z), 0, make_float3(dir.x,dir.y,dir.z), 1e34f };
					*second++ = r;
				}
			}
			
			// Next primary ray/hit
			primRay++;
			primHit++;
		}
	}  
}

void CopyOutputToTexture ( Buffer<Hit>& hitsBuffer, int w, int h, int spp )
{
	Hit* hits = hitsBuffer.ptr();

	// Retrieve hits from CUDA buffer
	std::vector<Hit> hitsTemp;	
	if( hitsBuffer.type() == RTP_BUFFER_TYPE_CUDA_LINEAR ) {		
		hitsTemp.resize(hitsBuffer.count());
		CHK_CUDA( cudaMemcpy( &hitsTemp[0], hitsBuffer.ptr(), hitsBuffer.sizeInBytes(), cudaMemcpyDeviceToHost ) );
		hits = &hitsTemp[0];
	}

	// Structure of hits buffer matches the ray buffer dimensions		
	unsigned char* pix = g_OPShadowData;	
	float v; 
	for (int y=0; y < h; y++ ) {
		for ( int x=0; x < w; x++ ) {			
			v = 0;
			for ( int s=0; s < spp; s++ ) {
				v += (hits->t < 0.0f) ? 1.0f : 0.0f;		
				hits++;
			}
			// BGRA format
			pix[2] = v * 255.0f / (float) spp;		// R = shadow value
			pix += 4;
		}
	}

	// Bind and write OpenGL texture for output
	glBindTexture( GL_TEXTURE_2D, g_OPTex );
	glPixelStorei( GL_UNPACK_ALIGNMENT, 1);
	glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA8, w, h, 0, GL_BGRA, GL_UNSIGNED_BYTE, g_OPShadowData );		
	glBindTexture( GL_TEXTURE_2D, 0 );
}

void ResizeBuffersOptixPrime ( int w, int h, int spp, RTPbuffertype bufferType )
{
	if ( spp != g_OPSamples ) {
		g_OPSamples = spp;		
		g_OPRays[0].free ();
		g_OPHits[0].free ();
		g_OPRays[1].free ();
		g_OPHits[1].free ();
		g_OPRays[0].alloc ( w*h, bufferType, LOCKED );			
		g_OPHits[0].alloc ( w*h, bufferType, LOCKED );
		g_OPRays[1].alloc ( spp*w*h, bufferType, LOCKED );
		g_OPHits[1].alloc ( spp*w*h, bufferType, LOCKED );
	}
}

void renderInitializeOptixPrime ( Scene& scene, int w, int h )
{
	// Create OptiX context
	RTPcontexttype contextType = RTP_CONTEXT_TYPE_CPU;
    contextType = RTP_CONTEXT_TYPE_CUDA;
	CHK_PRIME ( rtpContextCreate ( contextType, &g_OPContext ) );

	// Create Ray and Hit buffers 
	//  (populate and retrieve happens later on every frame)
	ResizeBuffersOptixPrime ( w, h, 1, RTP_BUFFER_TYPE_HOST );

	// Create intermediate host buffer for shadow results
	g_OPShadowData = (unsigned char*) malloc ( w * h * 4 );

	// Create an output texture for OpenGL
	glGenTextures( 1, (GLuint*) &g_OPTex );
	glBindTexture( GL_TEXTURE_2D, g_OPTex );	
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);		// Change these to GL_LINEAR for super- or sub-sampling
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);			
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);	// GL_CLAMP_TO_EDGE for linear filtering, not relevant for nearest.
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glBindTexture( GL_TEXTURE_2D, 0);
}


void renderAddModelOptixPrime ( Model* model ) 

{
	// Create geometry buffers
	RTPbufferdesc indicesDesc;
	CHK_PRIME ( rtpBufferDescCreate (
		g_OPContext, 
		RTP_BUFFER_FORMAT_INDICES_INT3,
		RTP_BUFFER_TYPE_HOST,
		model->elemBuffer,
		&indicesDesc)
		);

	CHK_PRIME( rtpBufferDescSetRange( indicesDesc, 0, model->elemCount ) );
	CHK_PRIME( rtpBufferDescSetStride( indicesDesc, model->elemStride ) );
  
	RTPbufferdesc verticesDesc;
	CHK_PRIME( rtpBufferDescCreate(
		g_OPContext,
		RTP_BUFFER_FORMAT_VERTEX_FLOAT3,
		RTP_BUFFER_TYPE_HOST,
		model->vertBuffer, 
		&verticesDesc )
		);
	CHK_PRIME( rtpBufferDescSetRange( verticesDesc, 0, model->vertCount ) );
	CHK_PRIME( rtpBufferDescSetStride( verticesDesc, model->vertStride ) );

	// Create model object
	RTPmodel op_model;
	CHK_PRIME( rtpModelCreate( g_OPContext, &op_model ) );
	CHK_PRIME( rtpModelSetTriangles( op_model, indicesDesc, verticesDesc ) );
	g_OPModels.push_back ( op_model );

	int useCallerTris=1;
	CHK_PRIME( rtpModelSetBuilderParameter( op_model, RTP_BUILDER_PARAM_USE_CALLER_TRIANGLES, sizeof(int), &useCallerTris ) );

	CHK_PRIME( rtpModelUpdate( op_model, RTP_MODEL_HINT_NONE ) );
}

Vector4DF renderOptixPrime ( Scene& scene, int w, int h, int spp ) 
{
	Vector4DF timing;
	RTPquery query;

	// Resize buffers if spp changes..
	if ( spp != g_OPSamples ) 
		ResizeBuffersOptixPrime ( w, h, spp, RTP_BUFFER_TYPE_HOST );		
	
	// Create buffer for ray input 	
	PERF_PUSH ( "Create primary" );
	CreatePrimaryRays ( g_OPRays[0], scene, w, h );
	timing.x = PERF_POP ();
	
	// Setup primary ray buffer
	CHK_PRIME( rtpBufferDescCreate( g_OPContext, Ray::format, g_OPRays[0].type(), g_OPRays[0].ptr(), &g_OPRaysDesc[0] )	);
	CHK_PRIME( rtpBufferDescSetRange( g_OPRaysDesc[0], 0, g_OPRays[0].count() ) );

	// Setup returned hit buffer
	CHK_PRIME( rtpBufferDescCreate( g_OPContext, Hit::format, g_OPHits[0].type(), g_OPHits[0].ptr(), &g_OPHitsDesc[0] )	);
	CHK_PRIME( rtpBufferDescSetRange( g_OPHitsDesc[0], 0, g_OPHits[0].count() ) );

	// Raytrace Primary rays
	PERF_PUSH ( "Raytace primary" );
	RTPmodel m = g_OPModels[0];
	CHK_PRIME( rtpQueryCreate( m, RTP_QUERY_TYPE_CLOSEST, &query ) );
	CHK_PRIME( rtpQuerySetRays( query, g_OPRaysDesc[0] ) );
	CHK_PRIME( rtpQuerySetHits( query, g_OPHitsDesc[0] ) );
	CHK_PRIME( rtpQueryExecute( query, 0 ) );
	timing.y = PERF_POP ();
	
	// Create secondary rays
	PERF_PUSH ( "Create shadow rays" );
	int3* modelElems = (int3*) scene.getModel(0)->elemBuffer;
	Vector3DF* modelVerts = (Vector3DF*) scene.getModel(0)->vertBuffer;
	CreateShadowRays ( g_OPRays[1], scene, w, h, g_OPRays[0], g_OPHits[0], spp, modelElems, modelVerts );
	timing.z = PERF_POP ();

	// Setup primary ray buffer
	CHK_PRIME( rtpBufferDescCreate( g_OPContext, Ray::format, g_OPRays[1].type(), g_OPRays[1].ptr(), &g_OPRaysDesc[1] )	);
	CHK_PRIME( rtpBufferDescSetRange( g_OPRaysDesc[1], 0, g_OPRays[1].count() ) );

	// Setup returned hit buffer
	CHK_PRIME( rtpBufferDescCreate( g_OPContext, Hit::format, g_OPHits[1].type(), g_OPHits[1].ptr(), &g_OPHitsDesc[1] )	);
	CHK_PRIME( rtpBufferDescSetRange( g_OPHitsDesc[1], 0, g_OPHits[1].count() ) );

	// Raytrace Secondary rays
	PERF_PUSH ( "Raytrace shadow rays" );	
	CHK_PRIME( rtpQueryCreate( m, RTP_QUERY_TYPE_ANY, &query ) );
	CHK_PRIME( rtpQuerySetRays( query, g_OPRaysDesc[1] ) );	
	CHK_PRIME( rtpQuerySetHits( query, g_OPHitsDesc[1] ) );
	CHK_PRIME( rtpQueryExecute( query, 0 ) );
	timing.w = PERF_POP ();

	PERF_PUSH ( "Write output" );
	CopyOutputToTexture ( g_OPHits[1], w, h, spp );
	PERF_POP ();

	return timing;
}

void renderSetupOptixPrimeGL ( Scene& scene, int prog, int w, int h )
{
	// OpenGL specific code to bind texture to the GLSL shader

	int sz[2] = { w, h };
	glProgramUniform2iv( prog, scene.getParam(prog, UOVERSIZE), 1, sz );     // Set value for "renderSize" uniform in the shader
	
	glProgramUniform1i ( prog, scene.getParam(prog, UOVERTEX), 0 );
	glActiveTexture ( GL_TEXTURE0 );
	glBindTexture ( GL_TEXTURE_2D, g_OPTex );
}


#endif