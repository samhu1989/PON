import bpy;

def parent_obj_to_camera(b_camera):
    origin = (0, 0, 0)
    b_empty = bpy.data.objects.new("Empty", None)
    b_empty.location = origin
    b_camera.parent = b_empty  # setup parenting

    scn = bpy.context.scene
    scn.objects.link(b_empty)
    scn.objects.active = b_empty
    return b_empty

scene = bpy.context.scene
scene.render.resolution_x = 224
scene.render.resolution_y = 224
scene.render.resolution_percentage = 100
scene.render.alpha_mode = 'TRANSPARENT'
cam = scene.objects['Camera']
cam.location = (0, 2.0, 1.5)
cam_constraint = cam.constraints.new(type='TRACK_TO')
cam_constraint.track_axis = 'TRACK_NEGATIVE_Z'
cam_constraint.up_axis = 'UP_Y'
b_empty = parent_obj_to_camera(cam)
cam_constraint.target = b_empty

mv_matrix = cam.matrix_world.inverted()
proj_matrix = cam.calc_matrix_camera(
            scene.render.resolution_x,
            scene.render.resolution_y,
            scene.render.pixel_aspect_x,
            scene.render.pixel_aspect_y,
            );
import numpy as np
np.set_printoptions(precision=7)
print(np.array(mv_matrix));
print(np.array(proj_matrix));
