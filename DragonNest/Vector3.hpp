#ifndef DN_VEC3_HPP
#define DN_VEC3_HPP

// Better compile with -O3 -ffast-math -mavx
// else use SSE only to support all CPUs.

// Replace D3DXVec3Normalize with Vec3Normalize* function
// to fix misaligned particle

// Example code assembly output: https://godbolt.org/z/nTMYo3q7G

// TODO: Fix MSVC missed branchless optimization on Vec3NormalizeBranchless with SSE
// TODO: inline assembly?

#include <cstdint>
#include <immintrin.h>
#include <float.h>

struct Vec3 { float x, y, z; };
// do not touch d3dx9 vec3normalize 32bit epsilon //or use 0.000000000000014210855F
static constexpr float epsilon = FLT_EPSILON; 
static constexpr float THREE = 3.0f;
static constexpr const float HALF = 0.5f;


// Branchless
inline Vec3* Vec3NormalizeBranchless(Vec3* out, const Vec3* in) {
	const float x = in->x;
	const float y = in->y;
	const float z = in->z;
	
	const float length_sq = x * x + y * y + z * z;
	
	const float mask = (length_sq < epsilon) ? 1.0f : 0.0f;
	
	const float rsqrt_approx = _mm_cvtss_f32(_mm_rsqrt_ss(_mm_set_ss(length_sq)));
	
	const float rsqrt_refined_unmasked = rsqrt_approx * (THREE * HALF - HALF * length_sq * rsqrt_approx * rsqrt_approx);
	
	const float rsqrt_refined = rsqrt_refined_unmasked * (1.0f - mask);

	out->x = x * rsqrt_refined;
	out->y = y * rsqrt_refined;
	out->z = z * rsqrt_refined;
	
	return out;
}

// Branch
inline Vec3* Vec3NormalizeBranch(Vec3* out, const Vec3* in) {
	const float x = in->x;
	const float y = in->y;
	const float z = in->z;
	
	const float length_sq = x * x + y * y + z * z;
	
	if (length_sq < epsilon) {
		out->x = 0.0f;
		out->y = 0.0f;
		out->z = 0.0f;
		return out;
	}
	
	const float rsqrt_approx = _mm_cvtss_f32(_mm_rsqrt_ss(_mm_set_ss(length_sq)));
	
	const float rsqrt_refined = rsqrt_approx * (THREE * HALF - HALF * length_sq * rsqrt_approx * rsqrt_approx);
	
	out->x = x * rsqrt_refined;
	out->y = y * rsqrt_refined;
	out->z = z * rsqrt_refined;
	
	return out;
}


#endif