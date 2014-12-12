
/*
 * Copyright (c) 2008 - 2009 NVIDIA Corporation.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property and proprietary
 * rights in and to this software, related documentation and any modifications thereto.
 * Any use, reproduction, disclosure or distribution of this software and related
 * documentation without an express license agreement from NVIDIA Corporation is strictly
 * prohibited.
 *
 * TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THIS SOFTWARE IS PROVIDED *AS IS*
 * AND NVIDIA AND ITS SUPPLIERS DISCLAIM ALL WARRANTIES, EITHER EXPRESS OR IMPLIED,
 * INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
 * PARTICULAR PURPOSE.  IN NO EVENT SHALL NVIDIA OR ITS SUPPLIERS BE LIABLE FOR ANY
 * SPECIAL, INCIDENTAL, INDIRECT, OR CONSEQUENTIAL DAMAGES WHATSOEVER (INCLUDING, WITHOUT
 * LIMITATION, DAMAGES FOR LOSS OF BUSINESS PROFITS, BUSINESS INTERRUPTION, LOSS OF
 * BUSINESS INFORMATION, OR ANY OTHER PECUNIARY LOSS) ARISING OUT OF THE USE OF OR
 * INABILITY TO USE THIS SOFTWARE, EVEN IF NVIDIA HAS BEEN ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGES
 */

#define WIN_TITLE	"OpenGL + Optix Compositing Demo (c) 2014, NVIDIA"

#include "app_util.h"
#include "app_perf.h"

#ifdef BUILD_OPTIX
	#include <optixu/optixpp_namespace.h>
	#include <optixu/optixu_math_namespace.h>
	#include <optixu/optixu_matrix_namespace.h>
	using namespace optix;
#endif

#include <string>
#include <iostream>
#include <stdlib.h>
#include <string.h>

#include "scene.h"
#include "render.h"

#define MODE_OPENGL		0	
#define MODE_OPTIX		1
#define MODE_OPTIXPRIME 2

#define MODE_CAMERA		0
#define MODE_FOV		1
#define MODE_LIGHT		2

int			draw_mode;
int			cam_mode;
int			frame = 0;
float		frameTime;
std::string	gTimingFile;
FILE*		gTimingFP;
int			num_samples = 1;

Scene		scene;

int			last_x, last_y;
int			dragging;


class AppWindow : public NVPWindow
{
	bool	m_validated;
public:
	AppWindow() : m_validated(false) {}
    virtual bool init();
    virtual void reshape(int w, int h);
    virtual void motion(int x, int y);
    virtual void mouse(NVPWindow::MouseButton button, ButtonAction action, int mods, int x, int y);
    virtual void keyboardchar(unsigned char key, int mods, int x, int y);
    virtual void display();
};


// Main display loop
void AppWindow::display () 
{
	float t[10];		// frame timings

	// Animation
	PERF_PUSH ( "animate" );
	scene.DoAnimation ( frame );
	PERF_POP ();

	switch ( draw_mode ) {
	case MODE_OPENGL: 
		
		t[0] = -1;	// no shadow timing

		// Basic OpenGL render. 
		PERF_PUSH ( "render" );
		
		renderCamSetupGL ( scene, GLS_PROGRAM );		
		renderLightSetupGL ( scene, GLS_PROGRAM );
		
		glShadeModel ( GL_FLAT );
		glClearColor ( 1, 1, 1, 1 );
		glClear ( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );
				
		// Use default shading program 
		glUseProgram ( GLS_PROGRAM );

		// Render polygons
		scene.SetMaterial ( -1, Vector4DF(1,1,1,1), Vector4DF(0.1,0.1,0.1,1), Vector4DF(1,1,1,1) );		
		renderSceneGL ( scene, GLS_PROGRAM );
		
		// Render outlines
		scene.SetOverrideMaterial ( Vector4DF(0.4,0.4,0.4,1), Vector4DF(0.1,0.1,0.1,1), Vector4DF(0,0,0,1) ); // black
		glPolygonMode ( GL_FRONT_AND_BACK, GL_LINE );	
		glEnable ( GL_POLYGON_OFFSET_LINE );
		glPolygonOffset ( 0, -0.2 );
		renderSceneGL ( scene, GLS_PROGRAM );
		glPolygonMode ( GL_FRONT_AND_BACK, GL_FILL );
		glDisable ( GL_POLYGON_OFFSET_LINE );

		glUseProgram ( 0);

		t[0] = PERF_POP ();

		break;

	case MODE_OPTIX: case MODE_OPTIXPRIME:
		
		#if defined(BUILD_OPTIX) || defined(BUILD_OPTIX_PRIME) 
		
			// Render shadows with either Optix or Optix Prime			
			PERF_PUSH ( "shadow" );			
			int finalshader;
			#ifdef BUILD_OPTIX
				if (draw_mode == MODE_OPTIX) {
					finalshader = OPTIX_PROGRAM; 
					t[0] = renderOptix ( scene, frame, num_samples );
					renderSetupOptixGL ( scene, finalshader );
				}
			#endif
			#ifdef BUILD_OPTIX_PRIME
				if (draw_mode == MODE_OPTIXPRIME) {
					finalshader = OPTIXPRIME_PROGRAM; 
					Vector4DF timing = renderOptixPrime ( scene, getWidth(), getHeight(), num_samples );
					t[0] = timing.w;
					t[2] = timing.y;					
					renderSetupOptixPrimeGL ( scene, finalshader, getWidth(), getHeight() );
				}
			#endif
			glFinish ();

			
			
			PERF_POP ();

			// Final render with OpenGL
			glClearColor ( 1, 1, 1, 1 );
			glClear ( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );

			renderCamSetupGL  ( scene, finalshader );		// Send camera params to final shader
			renderLightSetupGL  ( scene, finalshader );		// Send light params to final shader			
					
			PERF_PUSH ( "render" );
			glUseProgram ( finalshader );		
			scene.SetMaterial ( -1, Vector4DF(.2,.2,.2,1), Vector4DF(0.8,0.8,0.8,1), Vector4DF(1,1,1,1) );			
			renderSceneGL ( scene, finalshader );

			// Render outlines
			scene.SetOverrideMaterial ( Vector4DF(0.1,0.1,0.1,1), Vector4DF(0.4,0.4,0.4,1), Vector4DF(0,0,0,1) ); // black
			glPolygonMode ( GL_FRONT_AND_BACK, GL_LINE );	// Render outlines using override colors
			glEnable ( GL_POLYGON_OFFSET_LINE );
			glPolygonOffset ( 0, -0.2 );
			renderSceneGL ( scene, finalshader );
			glPolygonMode ( GL_FRONT_AND_BACK, GL_FILL );
			glDisable ( GL_POLYGON_OFFSET_LINE ); 

			t[1] = PERF_POP ();

			glUseProgram ( 0 );		

		#endif

		break;

		break;
	};

	//drawScene ();			// Draw scene 
	//drawGui ();			// Draw GUI (nvGUI in app_util)
	//draw2D ();			// Draw 2D (nv2D in app_util)

	// OpenGL - Swap buffers
	PERF_PUSH ( "present" );
	swapBuffers();
	t[3] = PERF_POP ();

	// Writing timing
	if ( gTimingFP != 0x0 ) {
		// frame #, animate, shadow, render, present
		fprintf ( gTimingFP, "%d, %f, %f, %f, %f\n", frame, t[0], t[1], t[2], t[3] );		
	}

	frame++;
}

void AppWindow::reshape ( int width, int height ) 
{
	// set window height and width
	setWinSz ( width, height );
	glViewport( 0, 0, width, height );  
	scene.SetAspect ( width, height );
	setview2D ( width, height );
	ResizeMSAA ();
}

// This function called by both OpenGL (GLUT) and DirectX
void AppWindow::keyboardchar (unsigned char key, int mods, int x, int y)
{
	switch ( key ) {		
	case ' ':	scene.RecordKeypoint ( getWidth(), getHeight() );	break;
	case 27:
				exit ( 0 ); 
				break;
	case 's':	
				num_samples = (num_samples == 1 ) ? 32 : 1;
				nvprintf  ( "Num samples: %d\n", num_samples );
				break;
	case 'l': case 'L':	cam_mode = MODE_LIGHT;	break;
	case 'c': case 'C': cam_mode = MODE_CAMERA; break;		
	case 'f': case 'F': cam_mode = MODE_FOV; break;		
	case '1':	draw_mode = MODE_OPENGL;	break;	
	case '2':	draw_mode = MODE_OPTIX;	break;		
	case '3':	draw_mode = MODE_OPTIXPRIME;	break;		
	};
}

void AppWindow::mouse ( NVPWindow::MouseButton button, ButtonAction action, int mods, int x, int y )
{
	Vector3DF cangs, ctp;
	float cdist;
	Camera3D* cam = scene.getCamera ();
	cangs = cam->getAng();
	ctp = cam->getToPos();
	cdist = cam->getOrbitDist();

	if ( action==NVPWindow::BUTTON_PRESS && guiMouseDown ( x, y ) ) return;

	if( action==NVPWindow::BUTTON_PRESS ) {		
		dragging = (int) button;		
		last_x = x;
		last_y = y;	
	} else if ( action==NVPWindow::BUTTON_RELEASE ) {
		dragging = -1;
	}
}

void AppWindow::motion ( int x, int y )
{
	int dx = last_x - x;
	int dy = y - last_y;

	// Handle GUI interaction in nvGui by calling guiMouseDrag
	if ( guiMouseDrag ( x, y ) ) return;

	// Camera interaction
	Camera3D* cam = scene.getCamera ();	
	switch ( cam_mode ) {
	case MODE_CAMERA:
		if ( dragging == NVPWindow::MOUSE_BUTTON_LEFT ) {
			Vector4DF angs = cam->getAng();
			angs.x += dx*.1;
			angs.y += dy*.1;
			cam->setOrbit ( angs, cam->getToPos(), cam->getOrbitDist(), cam->getOrbitDist() );
			//cam->moveOrbit ( dx*.1, dy*.1, 0, 0 );
		} else if ( dragging == NVPWindow::MOUSE_BUTTON_MIDDLE ) {
			cam->moveRelative ( dx*.1, dy*.1, 0 );		
		} else if ( dragging == NVPWindow::MOUSE_BUTTON_RIGHT ) {
			float orb = cam->getOrbitDist() + dy*0.1;
			cam->setOrbit ( cam->getAng(), cam->getToPos(), orb, orb);			
		} 
		break;
	case MODE_FOV:
		if ( dragging == NVPWindow::MOUSE_BUTTON_LEFT  ) {
			cam->setFov ( cam->getFov() + dy*.1 );
		}
		break;
	case MODE_LIGHT:				
		Light* light = scene.getLight();
		if ( dragging == NVPWindow::MOUSE_BUTTON_LEFT  ) {
			light->moveOrbit ( dx*0.1, dy*0.1, 0, 0 );
		} else if ( dragging == NVPWindow::MOUSE_BUTTON_RIGHT ) {
			light->moveOrbit ( 0, 0, dy*0.1, 0 );
		}
		break;	
	}
	last_x = x;
	last_y = y;	
}


bool AppWindow::init ()
{
	// Get comnmand line
	std::string str = "";  //cmdline;
	if ( str == "" )
		str = "-i lucy.scn -s 8";

	std::vector<std::string>	args;
	while ( str.length() > 0) {
		args.push_back ( strSplit ( str, " " ) );		
	}
	std::string filename = "";
	for (int n=0; n < args.size(); n++ ) {
		if ( args[n].compare ( "-p" ) == 0 ) {		// added path
			scene.AddPath ( std::string( args[n+1] ) );			
		}
		if ( args[n].compare ( "-i" ) == 0 ) {		// input file (scn)
			filename = args[n+1];
		}
		if ( args[n].compare ( "-t" ) == 0 ) {		// timing output
			gTimingFile = "timing.csv";
		}
		if ( args[n].compare ( "-s" ) == 0 ) {		// # samples
			num_samples = atoi ( args[n+1].c_str() );
		}
	}

	// Write timing file
	gTimingFP = 0x0;
	if ( gTimingFile != "" ) {
		char name[1024];
		strcpy ( name, gTimingFile.c_str() );
		gTimingFP = fopen ( name, "wt" );
	}

	nvprintf  ( "OpenGL + OptiX Compositing Demo\n" );
	nvprintf  ( "Copyright (c) 2014, NVIDIA Corporation\n" );	

	//-------- GUI
	addGui (  20,  20, 200, 24, "Frame Time (ms)",		GUI_PRINT,  GUI_FLOAT,	&frameTime, 0, 0 );

	//-------- Scene 
	
	// Create camera
	nvprintf  ( "Creating camera...\n" );
	Camera3D* cam = scene.AddCamera ();
	cam->setOrbit ( Vector3DF(45,30,0), Vector3DF(0,0,0), 120, 120 );
	cam->setNearFar ( 1, 1000 );
	cam->setFov ( 71.635 );
	cam->updateMatricies ();

	// Create model(s)
	nvprintf ( "Creating model(s)...\n" );
	nvprintf ( "  Project path: %s\n", std::string(PROJECT_ABSDIRECTORY).c_str() );
	scene.AddPath ( "..\\assets\\" );
	scene.AddPath ( "..\\shaders\\" );
	scene.AddPath ( std::string(PROJECT_RELDIRECTORY) + "\\assets\\" );
	scene.AddPath ( std::string(PROJECT_RELDIRECTORY) + "\\shaders\\" );
	scene.AddPath ( std::string(PROJECT_ABSDIRECTORY) + "\\assets\\" );
	scene.AddPath ( std::string(PROJECT_ABSDIRECTORY) + "\\shaders\\" );
	scene.LoadFile ( filename );
	
	// Initialize fonts
	/* init2D ( "data/arial_24" );
	setText ( 0.5, -0.5 );
	setview2D ( getWidth(), getHeight() );
	setorder2D ( true, -0.00001 );*/		

	//MoveWindow ( 5, 5, window_width, window_height );

	// Setup OpenGL default render
	nvprintf  ( "Creating OpenGL shader...\n" );
	renderAddShaderGL    ( scene, "render_GL.vert.glsl", "render_GL.frag.glsl" );

	// Setup OptiX
	#ifdef BUILD_OPTIX
		nvprintf  ( "Creating Optix shader..\n" );
		renderAddShaderOptix ( scene, "render_Optix.vert.glsl", "render_Optix.frag.glsl" );
		nvprintf  ( "Initializing Optix..\n" );
		renderInitializeOptix ( getWidth(), getHeight() );
		nvprintf  ( "Adding models to Optix..\n" );
		int mat_id = renderAddMaterialOptix ( scene, "optix_shadow_rays" );
		
		for (int n=0; n < scene.getNumModels(); n++ )
			renderAddModelOptix ( scene.getModel(n), mat_id );

		nvprintf  ( "Validating Optix..\n" );				
		renderValidateOptix ();
	#endif

	// Setup OptiX Prime
	#ifdef BUILD_OPTIX_PRIME
		nvprintf  ( "Creating Optix Prime shader..\n" );		
		renderAddShaderOptixPrime ( scene, "render_Optix.vert.glsl", "render_Optix.frag.glsl" );
		nvprintf  ( "Initializing Optix Prime..\n" );
		renderInitializeOptixPrime ( scene, getWidth(), getHeight() );
		nvprintf  ( "Adding models to Optix..\n" );		
		for (int n=0; n < 1; n++ )
			renderAddModelOptixPrime ( scene.getModel(n) );
	#endif

	draw_mode = MODE_OPENGL;
	#ifdef BUILD_OPTIX
		draw_mode = MODE_OPTIX;
	#else
		nvprintf ( "***** ERROR:\n" );
		nvprintf ( " OptiX 3.6.3 was not found.\n");
		nvprintf ( " Please specify CUDA_LOCATION variable for CUDA 5.5 during cmake generate step.\n" );
		nvprintf ( " Running sample with OpenGL only. No hard shadows will appear.\n");
		nvprintf ( "*****\n\n" );
	#endif
	cam_mode = MODE_CAMERA;
	
	PERF_INIT ( 64, true, false, false, 0, "" );		// 32/64bit, CPU?, GPU?, Cons out?, Level, Log file	

	return "OpenGL + OptiX Compositing";
}



int sample_main ( int argc, const char** argv )
{
	

	static AppWindow appWindow;

	NVPWindow::ContextFlags context (
		4,      //major;
		3,      //minor;
		false,   //core;
		16,      //MSAA;
		24,     //depth bits
		8,      //stencil bits

		true,   //debug;
		false,  //robust;
		false,  //forward;
	    NULL   //share;
    );
	appWindow.sysVisibleConsole();

	if ( !appWindow.create("OpenGL + OptiX Compositing", &context ) )
		return false;

	appWindow.makeContextCurrent();
	appWindow.swapInterval(0);	

	while( AppWindow::sysPollEvents(false) )
    {
		appWindow.postRedisplay ( 1 );		// ask window to refresh (will call display)
    }

	return true;
}


