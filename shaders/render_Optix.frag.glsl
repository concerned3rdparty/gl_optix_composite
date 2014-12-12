
#version 440 core
#extension GL_NV_gpu_shader5 : require

// Shading params
uniform vec3	uLightPos;			// light pos
uniform vec4    uClrAmb;
uniform vec4    uClrDiff;
uniform vec4    uClrSpec;	

uniform sampler2D uOverTex;
uniform ivec2	  uOverSize;

// Data from our vertex shader
in vec3			WorldPos;
in vec3			WorldNormal; 

out vec4 result;

void main()
{
	// Sample OptiX result
	float shadow = texture ( uOverTex, gl_FragCoord.xy / uOverSize ).x;	

	vec3 normal = normalize(cross ( dFdx(WorldPos), dFdy(WorldPos) ) );

	// Simple Shading
	vec3 toLight = normalize( uLightPos.xyz - WorldPos );	
	vec3 diffuse = uClrDiff.xyz * shadow * max(dot( toLight, normal ), 0.0);	
	
	result = uClrAmb + vec4(diffuse, 1);
}


