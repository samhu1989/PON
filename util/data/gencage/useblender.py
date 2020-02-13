import os;

def render(inobj):
    os.system('/home/blender/blender-2.79/blender -b --python blenderdn.py -- --obj %s --depth_scale 1.0'%inobj);