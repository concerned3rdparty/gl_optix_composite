
#include "app_util.h"
#include "scene.h"
#include "model.h"
#include "loader_ObjarReader.h" 
#include "loader_OBJReader.h" 

Scene* Scene::gScene = 0x0;
CallbackParser* Scene::gParse = 0x0;

Scene::Scene ()
{
	mCamera = 0x0;
	mNumPaths = 0;
	mOutFile = "out.scn";
	mOutModel = "";
	mOutFrame = 0;
	mOutCam = new Camera3D;
	mOutLight = new Light;
	mShadowParams.Set ( 1.0, 1.0, -5e-4f, 0.01 );   // Default FTIZB params

	// Scene and Parse must be global singletons because 
	// the callback parser accepts pointers-to-member function
	// which must be static. These static callbacks can only 
	// access the scene/parser through global variables. See: LoadModel
	gScene = this;
	gParse = new CallbackParser;
}
Scene::~Scene ()
{
	
}

void Scene::AddPath ( std::string path )
{
	if ( mNumPaths < MAX_PATHS ) {
		int n = mNumPaths;
		mSearchPaths[ n ] = (char*) malloc ( path.length() + 1 );
		strcpy ( mSearchPaths[n], path.c_str() );	
		mNumPaths++;
	}
}

int Scene::AddModel ( std::string filestr , float scale)
{
	char filename[1024];
	strncpy ( filename, filestr.c_str(), 1024 );

	Model* m = new Model;
	mModels.push_back ( m );
	

	if ( OBJARReader::isMyFile ( filename ) ) {
		// OBJAR File
		OBJARReader load_objar;
		load_objar.LoadFile ( m, filename, mSearchPaths, mNumPaths );		
	
	} else if ( OBJReader::isMyFile ( filename ) ) {
		// OBJ FIle
		OBJReader load_obj;
		load_obj.LoadFile ( m, filename, mSearchPaths, mNumPaths );				
	}
	
	// Rescale if desired
	m->Transform ( Vector3DF(0,0,0), Vector3DF(scale,scale,scale) );

	// Save name of model (for recording)
	char buf[32];
	sprintf ( buf, "%f", scale );
	mOutModel = " model " + std::string(filename) + " " + std::string(buf);

	return mModels.size()-1;
}

int	Scene::AddGround ( float hgt, float scale )
{
	Model* m = new Model;
	mModels.push_back ( m );

	if ( OBJReader::isMyFile ( "ground.obj" ) ) {		
		OBJReader load_obj;
		load_obj.LoadFile ( m, "ground.obj", mSearchPaths, mNumPaths );				
	}
	// Rescale if desired
	m->Transform ( Vector3DF(0,hgt,0), Vector3DF(scale,scale,scale) );

	return mModels.size()-1;
}

Camera3D* Scene::AddCamera ()
{
	mCamera = new Camera3D;
	return mCamera;
}
Light* Scene::AddLight ()
{
	Light* light = new Light;
	mLights.push_back ( light );
	return light;
}

// Read a shader file into a character string
char *ReadShaderSource( char *fileName )
{
	FILE *fp = fopen( fileName, "rb" );
	if (!fp) return NULL;
	fseek( fp, 0L, SEEK_END );
	long fileSz = ftell( fp );
	fseek( fp, 0L, SEEK_SET );
	char *buf = (char *) malloc( fileSz+1 );
	if (!buf) { fclose(fp); return NULL; }
	fread( buf, 1, fileSz, fp );
	buf[fileSz] = '\0';
	fclose( fp );
	return buf;
}


// Create a GLSL program object from vertex and fragment shader files
int Scene::AddShader (char* vertfile, char* fragfile )
{
	int maxLog = 65536, lenLog;
	char log[65536];

	// Search paths
	char vertpath[1024];
	char fragpath[1024];
	if ( !LocateFile ( vertfile, vertpath, mSearchPaths, mNumPaths ) ) {
		nvprintf ( "ERROR: Unable to open '%s'\n", vertfile ); 
		nverror ();
	}
	if ( !LocateFile ( fragfile, fragpath, mSearchPaths, mNumPaths ) ) {
		nvprintf ( "ERROR: Unable to open '%s'\n", fragfile ); 
		nverror ();
	}

	// Read shader source	
	GLuint program = glCreateProgram();
    char *vertSource = ReadShaderSource( vertpath );
	if ( !vertSource ) {
		nvprintf ( "ERROR: Unable to read source '%s'\n", vertfile ); 
		nverror();
	}
	char *fragSource = ReadShaderSource( fragpath );
	if ( !fragSource) { 
		nvprintf ( "ERROR: Unable to read source '%s'\n", fragfile ); 
		nverror();
	}

	int statusOK;
	GLuint vShader = glCreateShader( GL_VERTEX_SHADER );
	glShaderSource( vShader, 1, (const GLchar**) &vertSource, NULL );
	glCompileShader( vShader );
	glGetShaderiv( vShader, GL_COMPILE_STATUS, &statusOK );
	if (!statusOK) { 
		glGetShaderInfoLog ( vShader, maxLog, &lenLog, log );		
		nvprintf ("***Compile Error in '%s'!\n", vertfile); 
		nvprintf ("  %s\n", log );		
		nverror ();
	}
	free( vertSource );

	GLuint fShader = glCreateShader( GL_FRAGMENT_SHADER );
	glShaderSource( fShader, 1, (const GLchar**) &fragSource, NULL );
	glCompileShader( fShader );
	glGetShaderiv( fShader, GL_COMPILE_STATUS, &statusOK );
	if (!statusOK) { 
		glGetShaderInfoLog ( fShader, maxLog, &lenLog, log );		
		nvprintf ("***Compile Error in '%s'!\n", fragfile); 
		nvprintf ("  %s\n", log );	
		nverror ();
	}	
	free( fragSource );

	glAttachShader( program, vShader );
	glAttachShader( program, fShader );
    glLinkProgram( program );
    glGetProgramiv( program, GL_LINK_STATUS, &statusOK );
    if ( !statusOK ) { 
		printf("***Error! Failed to link '%s' and '%s'!\n", vertfile, fragfile ); 
		nverror ();
	}

	mShaders.push_back ( program );
	mParams.push_back ( ParamList() );
	mProgToSlot[ program ] = mParams.size()-1;

    return program;
}

int	Scene::getSlot ( int prog )
{
	// Get abstract slot from a program ID
	for (int n=0; n < mShaders.size(); n++ ) {
		if ( mShaders[n]==prog )
			return n;
	}
	return -1;
}

int	Scene::AddParam ( int prog, int id, char* name )
{
	int ndx = glGetProgramResourceIndex ( prog, GL_UNIFORM, name );	
	int slot = getSlot ( prog );
	if ( slot != -1 )
		mParams[slot].p[id] = ndx;

	return ndx;
}

void Scene::SetAspect ( int w, int h )
{
	if ( mCamera != 0x0 ) {
		mCamera->setAspect ( (float) w / (float) h );
	}
}

int Scene::AddMaterial ()
{
	Mat m;
	m.id = mMaterials.size();
	mMaterials.push_back ( m );
	return m.id;
}

void Scene::SetMaterialParam ( int id, int p, Vector3DF val )
{
	if ( id < mMaterials.size() ) {
		mMaterials[id].mParam[p] = val;
	}
}


void Scene::SetMaterial ( int model, Vector4DF amb, Vector4DF diff, Vector4DF spec )
{
	clrOverride = false;
	if ( model == -1 ) {
		for (int n=0; n < mModels.size(); n++) {
			mModels[n]->clrAmb = amb;
			mModels[n]->clrDiff = diff;
			mModels[n]->clrSpec = spec;		
		}
	}
	if ( model >= 0 && model < mModels.size() ) {
		mModels[model]->clrAmb = amb;
		mModels[model]->clrDiff = diff;
		mModels[model]->clrSpec = spec;		
	}
}

void Scene::SetOverrideMaterial ( Vector4DF amb, Vector4DF diff, Vector4DF spec )
{
	clrOverride = true;
	clrAmb = amb;
	clrDiff = diff;
	clrSpec = spec;
}

void Scene::LoadPath () 
{
	char path[512];   
	gParse->GetToken( path );
	gScene->AddPath ( path );	
}

void Scene::LoadSize ()
{
	std::string line = gParse->ReadNextLine( false );
	std::string var, value;
	int w, h;

	size_t pos = line.find_first_of ( ':' );
	while ( pos != std::string::npos ) {
		var = line.substr ( 0, pos );
		value = line.substr ( pos+1 );	
		if ( var == "width" ) w = strToNum ( value );
		if ( var == "height" ) h = strToNum ( value );
		line = gParse->ReadNextLine(false);
		pos = line.find_first_of ( ':' );
	}	
	// Resize 
	// NOTE: We do not call reshape because the scene
	// is loaded before the context is ready. The
	// first reshape will happen later using these.

	/*window_width = w;
	window_height = h;*/
}

void Scene::LoadModel () 
{
	char modelFile[512];   
	char scale[64];
	gParse->GetToken( modelFile );
	gParse->GetToken( scale );	
	gScene->AddModel ( modelFile, strToNum( scale ) );
}
void Scene::LoadGround () 
{
	char hgt[64];	
	char scale[64];	
	gParse->GetToken( hgt );	
	gParse->GetToken( scale );	
	gScene->AddGround ( strToNum( hgt ), strToNum( scale ) );
}

void Scene::LoadCamera () 
{
	std::string line = gParse->ReadNextLine( false );
	std::string var, value;

	Camera3D* cam = gScene->AddCamera ();

	size_t pos = line.find_first_of ( ':' );
	while ( pos != std::string::npos ) {
		var = line.substr ( 0, pos );
		value = line.substr ( pos+1 );		
		gScene->UpdateValue ( 'C', 0, strToID(var), strToVec3(value) );		
		line = gParse->ReadNextLine(false);
		pos = line.find_first_of ( ':' );
	}	
}

void Scene::LoadLight () 
{
	std::string line = gParse->ReadNextLine( false );
	std::string var, value;

	Light* light = gScene->AddLight ();

	size_t pos = line.find_first_of ( ':' );
	while ( pos != std::string::npos ) {
		var = line.substr ( 0, pos );
		value = line.substr ( pos+1 );
		gScene->UpdateValue ( 'L', 0, strToID(var), strToVec3(value) );			
		line = gParse->ReadNextLine(false);
		pos = line.find_first_of ( ':' );
	}	
}

void Scene::LoadShadow ()
{
	std::string line = gParse->ReadNextLine( false );
	std::string var, value;
	size_t pos = line.find_first_of ( ':' );
	while ( pos != std::string::npos ) {
		var = line.substr ( 0, pos );
		value = line.substr ( pos+1 );
		gScene->UpdateValue ( 'S', 0, strToID(var), strToVec3(value) );			
		line = gParse->ReadNextLine(false);
		pos = line.find_first_of ( ':' );
	}	
}

void Scene::LoadAnimation ()
{
	// Token contains start and end frames
	char buf[512];
	gParse->GetToken ( buf );
	Vector3DF frames;

	std::string line = gParse->ReadNextLine ( false );
	std::string var, obj, str1, str2;
	Vector3DF val1, val2;

	size_t pos = line.find_first_of ( ':' );
	while ( pos != std::string::npos ) {		
		var = line.substr( 0, pos );
		if ( var=="frames" ) {
			str1 = line.substr ( pos+1 );
			frames = strToVec3 ( str1 );	
		} else {
			obj = strParse ( var, "(", ")" );	// read variable and object
			if ( obj != "" ) {
				str2 = line.substr ( pos+1 );		
				str1 = strSplit ( str2, "," );		// read start and end values
				val1 = strToVec3 ( str1 );
				val2 = strToVec3 ( str2 );			// convert values to vec3
				gScene->AddKey ( obj, var, frames.x, frames.y, val1, val2 );
			}
		}

		line = gParse->ReadNextLine(false);
		pos = line.find_first_of ( ':' );
	}
}

void Scene::AddKey ( std::string obj, std::string var, int f1, int f2, Vector3DF val1, Vector3DF val2 )
{
	Key k;
	k.f1 = (float) f1;
	k.f2 = f2;
	k.val1 = val1;
	k.val2 = val2;
	k.obj = obj.at(0);
	k.objid = 0;
	k.varid = strToID ( var );

	mKeys.push_back ( k );
}

void Scene::DoAnimation ( int frame )
{
	Vector3DF val;
	float u;

	for (int n=0; n < mKeys.size(); n++ ) {	
		if ( frame >= mKeys[n].f1 && frame <= mKeys[n].f2 ) {
			u = (frame-mKeys[n].f1) / (mKeys[n].f2 - mKeys[n].f1);
			val = mKeys[n].val2; val -= mKeys[n].val1; val *= u;
			val += mKeys[n].val1;
			UpdateValue ( mKeys[n].obj, mKeys[n].objid, mKeys[n].varid, val );
		}
	}
}

void Scene::UpdateValue ( char obj, int objid, long varid, Vector3DF val )
{
	if ( obj=='C' ) {
		// Camera			
		Camera3D* cam = getCamera();
		switch ( varid ) {
		case 'look': cam->setToPos ( val.x, val.y, val.z );		break;
		case 'eye ': cam->setPos ( val.x, val.y, val.z );		break;		
		case 'near': cam->setNearFar ( val.x, cam->getFar());	break;
		case 'far ': cam->setNearFar ( cam->getNear(), val.x );	break;				
		case 'fov ': cam->setFov ( val.x );						break;
		case 'dist': cam->setDist ( val.x );					break;
		case 'angs': cam->setOrbit ( val, cam->getToPos(), cam->getOrbitDist(), cam->getOrbitDist() );	break;		
		};
	} else if ( obj=='L' ) {
		// Light
		Light* light = getLight();
		switch ( varid ) {
		case 'look': light->setToPos ( val.x, val.y, val.z );	break;
		case 'pos ': light->setPos ( val.x, val.y, val.z );		break;
		case 'near': light->setNearFar ( val.x, light->getFar());	break;
		case 'far ': light->setNearFar ( light->getNear(), val.x );	break;		
		case 'fov ': light->setFov ( val.x );					break;
		case 'dist': light->setDist ( val.x );					break;
		case 'angs': light->setOrbit ( val, light->getToPos(), light->getOrbitDist(), light->getOrbitDist() );	break;		
		};			
	} else if ( obj=='S' ) {
		// Shadows
		switch ( varid ) {
		case 'lamb': mShadowParams.x = val.x;	break;
		case 'dila': mShadowParams.y = val.x;	break;
		case 'epsi': mShadowParams.z = val.x;	break;
		case 'over': mShadowParams.w = val.x;	break;
		};
	}
}

void Scene::LoadFile ( std::string filestr  )
{
	char filename[1024];
	strncpy ( filename, filestr.c_str(), 1024 ); 

	nvprintf ( "Loading '%s'...\n", filename );

	// Set keywords & corresponding callbacks to process the data
	gParse->SetCallback( "path",            &Scene::LoadPath );
	gParse->SetCallback( "size",            &Scene::LoadSize );
	gParse->SetCallback( "model",           &Scene::LoadModel );
	gParse->SetCallback( "ground",			&Scene::LoadGround );
	gParse->SetCallback( "camera",			&Scene::LoadCamera );	
	gParse->SetCallback( "light",			&Scene::LoadLight );
	gParse->SetCallback( "animate",			&Scene::LoadAnimation );
	gParse->SetCallback( "shadow",			&Scene::LoadShadow );

	/*gParse->SetCallback( "addgroundplane",  &Scene::AddGroundPlane );
	gParse->SetCallback( "camera",          LoadCameraView );
	gParse->SetCallback( "cameraat",        LoadCameraAt );
	gParse->SetCallback( "light",           LoadLight );
	gParse->SetCallback( "pointlight",      LoadLight );
	gParse->SetCallback( "dirlight",        LoadDirectionalLight );
	gParse->SetCallback( "group",           NewGroup );
	gParse->SetCallback( "end",             EndGroup );
	gParse->SetCallback( "scale",           Scale );
	gParse->SetCallback( "translate",       Translate );
	gParse->SetCallback( "rotate",          Rotate );
	gParse->SetCallback( "matrix",          LoadMatrix );
	gParse->SetCallback( "movescale",       LoadMoveScale );*/

	// Go ahead and parse the file
	gParse->ParseFile ( filename, mSearchPaths, mNumPaths );
}

void Scene::RecordKeypoint ( int w, int h )
{
	int frame_delta = 500;

	char fname[512];
	strcpy ( fname, mOutFile.c_str() );	
	Camera3D* cam = getCamera ();
	Light* light = getLight ();

	if ( mOutFrame == 0 ) {
		// First keypoint - Write camera & light			
		FILE* fp = fopen ( fname, "wt" );
		fprintf ( fp, "\n" );
		fprintf ( fp, "size\n" );
		fprintf ( fp, "  width: %d\n", w );
		fprintf ( fp, "  height: %d\n", h );
		fprintf ( fp, "\n" );
		fprintf ( fp, " %s\n", mOutModel.c_str() );
		fprintf ( fp, "\n" );
		fprintf ( fp, "camera\n" );
		fprintf ( fp, "  look: %4.3f %4.3f %4.3f\n", cam->getToPos().x, cam->getToPos().y, cam->getToPos().z );
		fprintf ( fp, "  dist: %4.3f\n", cam->getOrbitDist() );
		fprintf ( fp, "  angs: %4.3f %4.3f %4.3f\n", cam->getAng().x, cam->getAng().y, cam->getAng().z );
		fprintf ( fp, "  fov:  %4.3f\n", cam->getFov() );
		fprintf ( fp, "  near: %5.5f\n", cam->getNear() );
		fprintf ( fp, "  far:  %5.5f\n", cam->getFar() );
		fprintf ( fp, "\n\n");
		fprintf ( fp, "light\n" );
		fprintf ( fp, "  look: %4.3f %4.3f %4.3f\n", light->getToPos().x, light->getToPos().y, light->getToPos().z );		
		fprintf ( fp, "  dist: %4.3f\n", light->getOrbitDist() );
		fprintf ( fp, "  angs: %4.3f %4.3f %4.3f\n", light->getAng().x, light->getAng().y, light->getAng().z );
		fprintf ( fp, "  fov:  %4.3f\n", light->getFov() );
		fprintf ( fp, "  near: %5.5f\n", light->getNear() );
		fprintf ( fp, "  far:  %5.5f\n", light->getFar() );
		fprintf ( fp, "\n\n");
	} else {
		// Later keypoints - Write animation
		FILE* fp = fopen ( fname, "a+t" );
		fprintf ( fp, "animate\n" );
		fprintf ( fp, "  frames: %d %d\n", mOutFrame, mOutFrame + frame_delta );
		fprintf ( fp, "  look (C): %4.3f %4.3f %4.3f, %4.3f %4.3f %4.3f\n", mOutCam->getToPos().x, mOutCam->getToPos().y, mOutCam->getToPos().z, cam->getToPos().x, cam->getToPos().y, cam->getToPos().z );
		fprintf ( fp, "  dist (C): %4.3f, %4.3f\n", mOutCam->getOrbitDist(), cam->getOrbitDist() );
		fprintf ( fp, "  angs (C): %4.3f %4.3f %4.3f, %4.3f %4.3f %4.3f\n", mOutCam->getAng().x, mOutCam->getAng().y,mOutCam->getAng().z, cam->getAng().x, cam->getAng().y, cam->getAng().z );
		fprintf ( fp, "  fov (C):  %4.3f, %4.3f\n", mOutCam->getFov(), cam->getFov() );				
		fprintf ( fp, "  look (L): %4.3f %4.3f %4.3f, %4.3f %4.3f %4.3f\n", mOutLight->getToPos().x, mOutLight->getToPos().y, mOutLight->getToPos().z,light->getToPos().x, light->getToPos().y, light->getToPos().z );		
		fprintf ( fp, "  dist (L): %4.3f, %4.3f\n", mOutLight->getOrbitDist(), light->getOrbitDist() );
		fprintf ( fp, "  angs (L): %4.3f %4.3f %4.3f, %4.3f %4.3f %4.3f\n", mOutLight->getAng().x, mOutLight->getAng().y,mOutLight->getAng().z, light->getAng().x, light->getAng().y, light->getAng().z );
		fprintf ( fp, "  fov (L):  %4.3f, %4.3f\n", mOutLight->getFov(), light->getFov() );		
		fprintf ( fp, "\n\n");	
		
		mOutFrame += frame_delta;
	}
	mOutCam->Copy ( *cam );
	mOutLight->Copy ( *light );	
}
