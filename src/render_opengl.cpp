
#include <GL/glew.h>

#include "scene.h"
#include "app_util.h"
#include "render.h"

int GLS_PROGRAM = -1;	// GL Shader Program

void renderAddShaderGL ( Scene& scene, char* vertname, char* fragname )
{
	GLS_PROGRAM = scene.AddShader ( vertname, fragname );
	scene.AddParam ( GLS_PROGRAM, UVIEW, "uView" );
	scene.AddParam ( GLS_PROGRAM, UPROJ, "uProj" );
	scene.AddParam ( GLS_PROGRAM, UMODEL, "uModel" );
	scene.AddParam ( GLS_PROGRAM, ULIGHTPOS, "uLightPos" );
	scene.AddParam ( GLS_PROGRAM, UCLRAMB, "uClrAmb" );
	scene.AddParam ( GLS_PROGRAM, UCLRDIFF, "uClrDiff" );
	scene.AddParam ( GLS_PROGRAM, UCLRSPEC, "uClrSpec" );	
}

void renderCamSetupGL ( Scene& scene, int prog )
{
	// Set model, view, projection matrices
	Camera3D* cam = scene.getCamera ();	
	Matrix4F ident;
	ident.Identity();	
	glProgramUniformMatrix4fv( prog, scene.getParam(prog, UMODEL), 1, GL_FALSE, ident.GetDataF() );
	glProgramUniformMatrix4fv( prog, scene.getParam(prog, UVIEW), 1, GL_FALSE, cam->getViewMatrix().GetDataF() ); 
	glProgramUniformMatrix4fv( prog, scene.getParam(prog, UPROJ), 1, GL_FALSE, cam->getProjMatrix().GetDataF() );
}

void renderLightSetupGL ( Scene& scene, int prog ) 
{
	Light* light = scene.getLight ();	
	glProgramUniform3fv  ( prog, scene.getParam(prog, ULIGHTPOS), 1, &light->getPos().x );	
}
void renderSetMaterialGL ( Scene& scene, int prog, Vector4DF amb, Vector4DF diff, Vector4DF spec )
{
	glProgramUniform4fv  ( prog, scene.getParam(prog, UCLRAMB),  1, &amb.x );
	glProgramUniform4fv  ( prog, scene.getParam(prog, UCLRDIFF), 1, &diff.x );
	glProgramUniform4fv  ( prog, scene.getParam(prog, UCLRSPEC), 1, &spec.x );
}

void renderSceneGL ( Scene& scene, int prog )
{
	glEnable ( GL_CULL_FACE );
	glEnable ( GL_DEPTH_TEST );		

	// Render each model
	Model* model;
	if ( scene.useOverride() )
		renderSetMaterialGL ( scene, prog, scene.clrAmb, scene.clrDiff, scene.clrSpec );

	for (int n = 0; n < scene.getNumModels(); n++ ) {
		model = scene.getModel( n );		
		if ( !scene.useOverride() ) renderSetMaterialGL ( scene, prog, model->clrAmb, model->clrDiff, model->clrSpec );
		glBindVertexArray ( model->vertArrayID );
		glBindBuffer ( GL_ELEMENT_ARRAY_BUFFER, model->elemBufferID );
		glDrawElements ( GL_TRIANGLES, model->elemCount * 3, GL_UNSIGNED_INT, 0 );
	}
	glBindBuffer ( GL_ELEMENT_ARRAY_BUFFER, 0 );
	glBindVertexArray ( 0 );
	glDisable ( GL_DEPTH_TEST );
	glDisable ( GL_CULL_FACE );	

	// OpenGL 3.3
	/* glEnableClientState ( GL_VERTEX_ARRAY );
	glBindBuffer ( GL_ARRAY_BUFFER, model->vertBufferID );	
	glVertexPointer ( model->vertComponents, GL_FLOAT, model->vertStride, (char*) model->vertOffset );
	glNormalPointer ( GL_FLOAT, model->vertStride, (char*) model->normOffset );
	glBindBuffer ( GL_ELEMENT_ARRAY_BUFFER, model->elemBufferID );	 
	glDrawElements ( model->elemDataType, model->elemCount*3, GL_UNSIGNED_INT, 0 );	*/
}