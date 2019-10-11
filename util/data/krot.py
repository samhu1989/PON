def importParamBIN(origin_list, lookat_list, upvec_list):
	paramRotList = list()
	paramTransList = list()
	cutList = list()
	
	x0 = -10000
	y0 = -10000
	x1 = 10000
	y1 = 10000
	
	origin = np.array([eval(i) for i in origin_list.split(',')])
	lookat = np.array([eval(i) for i in lookat_list.split(',')])
	viewUp = np.array([eval(i) for i in upvec_list.split(',')])
	
	viewDir = origin - lookat
	viewDir = viewDir / np.linalg.norm(viewDir)
	viewRight = np.cross(viewUp, viewDir)
	viewRight= viewRight / np.linalg.norm(viewRight)
	viewUp = np.cross(viewDir, viewRight)
	viewUp = viewUp / np.linalg.norm(viewUp)
	
	R = np.ndarray((3, 3))
	R[0, 0] = viewRight[0]
	R[1, 0] = viewRight[1]
	R[2, 0] = viewRight[2]
	R[0, 1] = viewUp[0]
	R[1, 1] = viewUp[1]
	R[2, 1] = viewUp[2]
	R[0, 2] = viewDir[0]
	R[1, 2] = viewDir[1]
	R[2, 2] = viewDir[2]
	R = inv(R);
        
	paramRotList.append(R)
	
	T = np.ndarray((3, 1))
	T[0, 0] = origin[0]
	T[1, 0] = origin[1]
	T[2, 0] = origin[2]
	T = np.dot(-R, T)
	
	paramTransList.append(T)
	
	cutList.append([x0, y0, x1, y1]);
	
	return (paramRotList, paramTransList, cutList)
    
cam_ob = bpy.context.scene.camera;
