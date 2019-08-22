import os;
pth = os.path.dirname(os.path.abspath(__file__));
def run(**kwargs):
    data_root = kwargs['data_path'];
    train = data_root + os.sep + 'train';
    val = data_root + os.sep + 'val';
    fs = os.listdir(train);
    for f in fs:
        if f.endswith('mesh.obj'):
            view_path = train+os.sep+f.replace('mesh.obj','view');
            os.mkdir(view_path);
            cmd = 'blender --background --python %s -- --output_folder %s --views 16 --obj %s'%(pth+os.sep+'render_blender.py',os.path.abspath(view_path),train+os.sep+f);
            os.system(cmd);
    fs = os.listdir(val);
    for f in fs:
        if f.endswith('mesh.obj'):
            view_path = val+os.sep+f.replace('mesh.obj','view');
            os.mkdir(view_path);
            cmd = 'blender --background --python %s -- --output_folder %s --views 16 --obj %s'%(pth+os.sep+'render_blender.py',os.path.abspath(view_path),val+os.sep+f);
            os.system(cmd);