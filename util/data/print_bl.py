import os;
pth = os.path.dirname(os.path.abspath(__file__));
def run(**kwargs):
    data_root = kwargs['data_path'];
    train = data_root + os.sep + 'train';
    val = data_root + os.sep + 'test';
    fs = os.listdir(val);
    for f in fs:
        if f.endswith('.ply'):
            view_path = val;
            cmd = 'blender --background --python %s -- --output_folder %s --views 1 --obj %s'%(pth+os.sep+'print_bl.py',os.path.abspath(view_path),val+os.sep+f);
            os.system(cmd);
            break;
