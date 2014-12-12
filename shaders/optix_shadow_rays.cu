
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

#include <optix.h>
#include <optixu/optixu_math_namespace.h>
#include "helpers.h"
#include "random.h"

using namespace optix;

rtDeclareVariable(float3,       light_pos, , );

rtDeclareVariable(rtObject,     top_object, , );
rtDeclareVariable(float,        scene_epsilon, , );
rtDeclareVariable(unsigned int, shadow_enable, , );
rtDeclareVariable(unsigned int, mirror_enable, , );
rtDeclareVariable(unsigned int, cone_enable, , );
rtDeclareVariable(int,          max_depth, , );
rtDeclareVariable(unsigned int, radiance_ray_type, , );
rtDeclareVariable(unsigned int, shadow_ray_type, , );
rtDeclareVariable(float3,		shading_normal, attribute shading_normal, ); 
rtDeclareVariable(float3,		front_hit_point, attribute front_hit_point, );
rtDeclareVariable(float3,		back_hit_point, attribute back_hit_point, );

rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );
rtDeclareVariable(float, t_hit, rtIntersectionDistance, );

rtDeclareVariable(float,        importance_cutoff, , );
rtDeclareVariable(float3,       cutoff_color, , );
rtDeclareVariable(int,          reflection_maxdepth, , );
rtDeclareVariable(uint2,        launch_index, rtLaunchIndex, );
rtDeclareVariable(unsigned int, frame_number, , );
rtDeclareVariable(unsigned int, num_samples, , );
rtBuffer<unsigned int, 2>        rnd_seeds;

struct PerRayData_radiance
{
  float3 result;
  float importance;
  int depth;
};

struct PerRayData_shadow
{
  float3 attenuation;
};

rtDeclareVariable(PerRayData_radiance, prd_radiance, rtPayload, );
rtDeclareVariable(PerRayData_shadow,   prd_shadow,   rtPayload, );

// -----------------------------------------------------------------------------

static __device__ __inline__ float3 TraceRay (float3 origin, float3 direction, int depth, float importance )
{
  optix::Ray ray = optix::make_Ray( origin, direction, radiance_ray_type, 0.0f, RT_DEFAULT_MAX );
  PerRayData_radiance prd;
  prd.depth = depth;
  prd.importance = importance;

  rtTrace( top_object, ray, prd );
  return prd.result;
}


static __device__ __inline__ float ShadowRay( float3 origin, float3 direction )
{
  optix::Ray ray = optix::make_Ray( origin, direction, shadow_ray_type, 0.0f, RT_DEFAULT_MAX );
  PerRayData_shadow prd;  
  prd.attenuation = make_float3(1.0f);
  rtTrace( top_object, ray, prd );

  return prd.attenuation.x;
}

static __device__ __inline__ float3 exp( const float3& x )
{
  return make_float3(exp(x.x), exp(x.y), exp(x.z));
}

// -----------------------------------------------------------------------------

float3 __device__ __inline__ jitter_sample ( const uint2& index, float3 pos, float3 amt )
{	 
    volatile unsigned int seed  = rnd_seeds[ index ]; // volatile workaround for cuda 2.0 bug
    unsigned int new_seed  = seed;
    float uu = rnd( new_seed )-0.5f;
    float vv = rnd( new_seed )-0.5f;
	float ww = rnd( new_seed )-0.5f;
    rnd_seeds[ index ] = new_seed;	
    return pos + amt*make_float3(uu,vv,ww);
}

RT_PROGRAM void closest_hit_radiance()
{
	// geometry vectors
	const float3 n = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, shading_normal)); // normal  
	const float3 fhp = rtTransformPoint(RT_OBJECT_TO_WORLD, front_hit_point);
	const float3 i = ray.direction;                                            // incident direction
	float3 L, P;
	float shadow; 

	if ( num_samples == 1 ) {
		shadow = ShadowRay ( fhp, normalize(light_pos - fhp) );
	} else {
		// approximation for screen coverage of pixel in world space
		float d = length ( fhp - ray.origin );
		float3 dxyz = make_float3(0.75) * d / 1280.0;
		L =	normalize ( light_pos - fhp );		// L = light direction
		for (int s=0; s < num_samples; s++ ) {
			// jitter sub-pixel samples at the hit point
			P = jitter_sample ( launch_index, fhp, dxyz ) + n*0.05f;
			//L =	normalize ( light_pos - P );		// L = light direction
			// add shadow rays
			shadow += ShadowRay ( P, L );
		}
		shadow *= 1.0f / num_samples;
	}
	prd_radiance.result = make_float3( shadow, 0, 0 );
}

// -----------------------------------------------------------------------------

//
// Attenuates shadow rays for shadowing transparent objects
//
rtDeclareVariable(float3, shadow_attenuation, , );

RT_PROGRAM void any_hit_shadow()
{
	prd_shadow.attenuation = make_float3(0);
}
