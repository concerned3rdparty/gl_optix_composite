
#include "scene.h"
#include "app_util.h"

#define UVIEW		0
#define UPROJ		1
#define UMODEL		2
#define ULIGHTPOS	3
#define UCLRAMB		4
#define UCLRDIFF	5
#define UCLRSPEC	6
#define USHADOWMASK	7
#define USHADOWSIZE	8
#define UEYELIGHT	9
#define UOVERTEX	10
#define UOVERSIZE	11

// OpenGL Render
extern int GLS_PROGRAM;
extern void renderAddShaderGL ( Scene& scene, char* vertfile, char* fragfile );
extern void renderCamSetupGL (  Scene& scene, int prog );
extern void renderLightSetupGL ( Scene& scene, int prog );
extern void renderSceneGL ( Scene& scene, int prog );

// FT-IZB Render
#ifdef BUILD_FTIZB
	extern int IZB_PROGRAM;
	extern void renderAddShaderIZB ( Scene& scene, char* vertname, char* fragname );
	extern void renderInitializeIZB ( Scene& scene, int w, int h );
	extern void renderAddModelIZB ( Model* model );
	extern void renderCamSetupIZB (  Scene& scene );
	extern void renderLightSetupIZB (  Scene& scene );
	extern void renderParamSetupIZB ( float l, float d, float e, float o);
	extern void renderMaskSetupIZB ( Scene& scene, int w, int h );
	extern void renderClearIZB ();
	extern void renderSetMaterialIZB ( Vector4DF amb, Vector4DF diff, Vector4DF spec );
	extern float renderShadowsIZB ( int spp, int cascades, bool timing );
#endif

// OptiX Render
#ifdef BUILD_OPTIX
	extern int OPTIX_PROGRAM;
	extern void renderAddShaderOptix ( Scene& scene, char* vertname, char* fragname );
	extern void renderInitializeOptix ( int w, int h );
	extern  int renderAddMaterialOptix ( Scene& scene, std::string name );
	extern void renderAddModelOptix ( Model* model, int oid );
	extern void renderValidateOptix ();
	extern float renderOptix ( Scene& scene, int frame, int spp );
	extern void renderSetupOptixGL ( Scene& scene, int prog );
#endif

// OptiX Prime Render
#ifdef BUILD_OPTIX_PRIME
	extern int OPTIXPRIME_PROGRAM;
	extern void renderAddShaderOptixPrime ( Scene& scene, char* vertname, char* fragname );
	extern void renderInitializeOptixPrime ( Scene& scene, int w, int h );
	extern void renderAddModelOptixPrime ( Model* model );
	extern Vector4DF renderOptixPrime ( Scene& scene, int w, int h, int spp );
	extern void renderSetupOptixPrimeGL ( Scene& scene, int prog, int w, int h );
#endif