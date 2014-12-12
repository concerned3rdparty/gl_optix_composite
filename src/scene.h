
#ifndef DEF_SCENE_H
	#define DEF_SCENE_H


	#include <vector>
	#include "model.h"
	#include "app_util.h"		// Camera3D

	#define MAX_PATHS	32
	
	struct ParamList {
		int p[128];
	};
	struct Key {
		char			obj;		// object type
		int				objid;		// object ID
		unsigned long	varid;		// variable ID
		float			f1, f2;		// frame start/end
		Vector3DF		val1, val2;	// value start/end		
	};
	struct Mat {
		int				id;		
		Vector3DF		mAmb, mDiff, mSpec;		
		Vector3DF		mParam[64];
	};
	class CallbackParser;

	class Scene {
	public:
		Scene();
		~Scene();
		static Scene*			gScene;
		static CallbackParser*	gParse;

		void		LoadFile ( std::string filename );		
		void		AddPath ( std::string path );
		Camera3D*	AddCamera ();
		Light*		AddLight ();
		int			AddModel ( std::string filename, float scale=1.0 );
		int			AddGround ( float hgt, float scale=1.0 );
		int			AddShader ( char* vertfile, char* fragfile );
		int			AddParam ( int prog, int id, char* name );
		void		AddKey ( std::string obj, std::string var, int f1, int f2, Vector3DF val1, Vector3DF val2 );
		void		SetAspect ( int w, int h );
		int			AddMaterial ();
		void		SetMaterial ( int model, Vector4DF amb, Vector4DF diff, Vector4DF spec );		
		void		SetMaterialParam ( int id, int p, Vector3DF val );
		void		SetOverrideMaterial ( Vector4DF amb, Vector4DF diff, Vector4DF spec );

		int			getNumModels ()		{ return mModels.size(); }
		Model*		getModel ( int n )	{ return mModels[n]; }
		Camera3D*	getCamera ()		{ return mCamera; }
		Light*		getLight ()			{ return mLights[0]; }
		int			getSlot ( int prog );
		int			getParam ( int prog, int id )	{ return mParams[ mProgToSlot[prog] ].p[id]; }
		bool		useOverride ()		{ return clrOverride; }
		Vector4DF	getShadowParams ()	{ return mShadowParams; }
		
		// Loading scenes
		void		Load ( char *filename, float windowAspect );
		static void	LoadPath ();
		static void LoadSize ();
		static void	LoadModel ();
		static void	LoadGround ();
		static void LoadCamera ();
		static void LoadLight ();
		static void LoadAnimation ();
		static void LoadShadow ();

		// Animation
		void		DoAnimation ( int frame );
		void		UpdateValue (  char obj, int objid, long varid, Vector3DF val );
		void		RecordKeypoint ( int w, int h );


	public:
		Camera3D*				mCamera;				
		std::vector<Model*>		mModels;
		std::vector<Light*>		mLights;
		std::vector<int>		mShaders;
		std::vector<ParamList>	mParams;
		std::vector<Key>		mKeys;
		std::vector<Mat>		mMaterials;

		bool		clrOverride;
		Vector4DF	clrAmb, clrDiff, clrSpec;	

		char*		mSearchPaths[MAX_PATHS];
		int			mNumPaths;

		int			mProgToSlot[ 512 ];
		
		// Animation recording
		std::string				mOutFile;
		int						mOutFrame;
		Camera3D*				mOutCam;
		Light*					mOutLight;	
		std::string				mOutModel;

		// Shadow parameters (independent of method used)
		Vector4DF				mShadowParams;
	};

#endif
