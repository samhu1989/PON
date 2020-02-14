import os;

def render(inobj):
    os.system('/home/blender/blender-2.79/blender -b --python blenderdn.py -- --obj %s'%inobj);
    
def render_msk(path,angle=0):
    alst = [];
    blst = [];
    for f in os.listdir(path):
        if '_a.ply' in f:
            alst.append(f);
        if '_b.ply' in f:
            blst.append(f);
    selfpath = os.path.join(path,'self_msk');
    if not os.path.exists(selfpath):
        os.mkdir(selfpath);
    allpath = os.path.join(path,'all_msk');
    if not os.path.exists(allpath):
        os.mkdir(allpath);
    for bf in blst:
        cmd = '/home/blender/blender-2.79/blender --background --python render_part.py'
        cmd += '-- --output_folder %s --views 1'%os.path.abspath(allpath)
        cmd += ' --objp %s'%os.path.abspath(path);
        lst = bf;
        for aa in alst:
            if aa == bf.replace('_b','_a'):
                continue;
            else:
                lst += '~' + aa;
        cmd += ' --objs %s'%lst;
        cmd += ' --angle %d'%angle;
        os.system(cmd);
    for bf in blst:
        cmd = '/home/blender/blender-2.79/blender --background --python render_part.py'
        cmd += '-- --output_folder %s --views 1'%os.path.abspath(selfpath)
        cmd += ' --objp %s'%os.path.abspath(path);
        lst = bf;
        cmd += ' --objs %s'%lst;
        cmd += ' --angle %d'%angle;
        os.system(cmd);
    return;