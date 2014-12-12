

#ifndef DEF_OPTIX_PRIME

	#include <optix_prime/optix_prime.h>
	#include <cuda_runtime.h>
	#include <stdlib.h>
	#include <iostream>
	#include <string>
	#include <vector>

	// Ray structure
	struct Ray
	{
		static const RTPbufferformat format = RTP_BUFFER_FORMAT_RAY_ORIGIN_TMIN_DIRECTION_TMAX;

		float3 origin;
		float  tmin;
		float3 dir;
		float  tmax;
	};
	
	// Hit structure
	struct Hit
	{
		static const RTPbufferformat format = RTP_BUFFER_FORMAT_HIT_T_TRIID_U_V;
		float t;
		int   triId;
		float u;
		float v;
	};
	
	enum PageLockedState {
	  UNLOCKED,
	  LOCKED
	};

	void CHK_CUDA( cudaError_t result ) {
		if ( result != cudaSuccess ) {
			std::cerr << "Error on line " << __LINE__ << ":" << cudaGetErrorString( result ) << std::endl;
			exit(1);
		}
	}


	// Buffer class
	template<typename T>
	class Buffer {
	public:
		Buffer( size_t count=0, RTPbuffertype type=RTP_BUFFER_TYPE_HOST, PageLockedState pageLockedState=UNLOCKED ) 
		: m_ptr( 0 )
		{
		alloc( count, type, pageLockedState );
		}

		void alloc( size_t count )	  {	alloc( count, m_type, m_pageLockedState );  }

		void alloc( size_t count, RTPbuffertype type, PageLockedState pageLockedState=UNLOCKED )
		{
			if ( m_ptr ) free();

			m_type = type;
			m_count = count;
			if( m_count > 0 ) {
				if( m_type == RTP_BUFFER_TYPE_HOST )  {
					m_ptr = new T[m_count];
					if( pageLockedState )
						rtpHostBufferLock( m_ptr, sizeInBytes() ); // for improved transfer performance
					m_pageLockedState = pageLockedState;
				} else {
					CHK_CUDA( cudaGetDevice( &m_device ) );
					CHK_CUDA( cudaMalloc( &m_ptr, sizeInBytes() ) );
				}
			}
		}

		void free() {
			if ( m_ptr == 0 ) return;
			if( m_ptr && m_type == RTP_BUFFER_TYPE_HOST ) {
				if( m_pageLockedState )
					rtpHostBufferUnlock( m_ptr );
				delete[] m_ptr;
			} else  {
				int oldDevice;
				CHK_CUDA( cudaGetDevice( &oldDevice ) );
				CHK_CUDA( cudaSetDevice( m_device ) );
				CHK_CUDA( cudaFree( m_ptr ) );
				CHK_CUDA( cudaSetDevice( oldDevice ) );
			}
			m_ptr = 0;
			m_count = 0;
		}

		~Buffer()  { free(); }

		size_t count()       { return m_count; }
		size_t sizeInBytes() { return m_count * sizeof(T); }
		T* ptr()             { return m_ptr; }
		RTPbuffertype type() { return m_type; }

	protected:
		RTPbuffertype m_type;
		T* m_ptr;
		int m_device;
		size_t m_count;
		PageLockedState m_pageLockedState;

	private:
		Buffer<T>( const Buffer<T>& );            // forbidden
		Buffer<T>& operator=( const Buffer<T>& ); // forbidden
	};

#endif