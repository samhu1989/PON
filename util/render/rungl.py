from util.dataset.ToyV import projm,mvm;
from util.data.gen_toybox import box_face;
import numpy as np;
from PIL import Image;
import platform as pf;
import os;
#Windows
if pf.platform().startswith('Windows'):
    import glfw;
    from OpenGL.GL import *
    from OpenGL import GL;
elif pf.platform().startswith('Linux'):
    os.environ['PYOPENGL_PLATFORM'] = 'osmesa';
    from OpenGL import osmesa;
    from OpenGL.GL import *
    from OpenGL import GL
else:
    assert(False,'Unkown platform');
    

w,h = 224,224;
box_vert = np.array(
    [
        [-0.5,0.5,0.0],[0.5,0.5,0.0],[0.5,-0.5,0.0],[-0.5,-0.5,0.0],
        [-0.5,0.5,0.5],[0.5,0.5,0.5],[0.5,-0.5,0.5],[-0.5,-0.5,0.5]
    ],dtype=np.float32);

def setDefaultMaterial():
    mat_a = [0.250000, 0.148000, 0.064750, 1.000000];
    mat_d = [0.400000, 0.236800, 0.103600, 1.000000];
    mat_s = [0.774597, 0.458561, 0.200621, 1.000000];
    shine = [76.800003];

    GL.glMaterialfv(GL.GL_FRONT_AND_BACK, GL.GL_AMBIENT,   mat_a);
    GL.glMaterialfv(GL.GL_FRONT_AND_BACK, GL.GL_DIFFUSE,   mat_d);
    GL.glMaterialfv(GL.GL_FRONT_AND_BACK, GL.GL_SPECULAR,  mat_s);
    GL.glMaterialfv(GL.GL_FRONT_AND_BACK, GL.GL_SHININESS, shine);
    return;

def setDefaultLight():
    pos1 = [ 4.07625,1.00545,5.90386,0.0];
    col1 = [ 0.98,  0.98,  0.98,  1.0];
    glEnable(GL_LIGHT0);
    glLightModeli(GL_LIGHT_MODEL_TWO_SIDE,GL_TRUE);    
    glLightfv(GL_LIGHT0,GL_POSITION, pos1);
    glLightfv(GL_LIGHT0,GL_DIFFUSE,  col1);
    glLightfv(GL_LIGHT0,GL_SPECULAR, col1);
    
def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0: 
       return v
    return v / norm;

def calnormal(p1,p2,p3):
    N = np.cross(p3-p1, p2-p1);
    return normalize(N);
    
def draw():
    glBegin(GL_TRIANGLES);
    for i in range(box_face.shape[0]):
        p = box_vert[box_face[i,:],:];
        n = calnormal(p[0,:].copy(),p[1,:].copy(),p[2,:].copy());
        glNormal3f(n[0],n[1],n[2]);
        for j in range(p.shape[0]):
            glVertex3f(p[j,0],p[j,1],p[j,2]);
    glEnd();
    glFlush();

def display(draw=draw):
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glMultMatrixd(mvm.T);
    draw();
    
def initGL():
    glViewport(0,0,w,h);
    glClearColor(1.0, 1.0, 1.0, 0.0); 
    glDisable( GL_DITHER );
    glEnable(GL_LIGHTING);
    glShadeModel(GL_FLAT);
    glEnable( GL_DEPTH_TEST );
    glLoadIdentity();
    setDefaultLight();
    glEnable(GL_LINE_SMOOTH);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glHint(GL_LINE_SMOOTH_HINT, GL_NICEST);

def reshape(width,height):
   if height == 0:
       height = 1;
   glViewport(0, 0, width, height);
   glMatrixMode(GL_PROJECTION);  
   glLoadIdentity(); 
   glMultMatrixd(projm.T);
   glMatrixMode(GL_MODELVIEW);

def myglCreateBuffers(width, height):
    fbo = glGenFramebuffers(1)
    color_buf = glGenRenderbuffers(1)
    depth_buf = glGenRenderbuffers(1)

    # binds created FBO to context both for read and draw
    glBindFramebuffer(GL_FRAMEBUFFER, fbo)

    # bind color render buffer
    glBindRenderbuffer(GL_RENDERBUFFER, color_buf)
    glRenderbufferStorage(GL_RENDERBUFFER, GL_RGBA8, width, height)
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, color_buf)

    # bind depth render buffer - no need for 2D, but necessary for real 3D rendering
    glBindRenderbuffer(GL_RENDERBUFFER, depth_buf)
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, width, height)
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, depth_buf)
    return fbo, color_buf, depth_buf, width, height
    
def myglReadColorBuffer(buffers):
    fbo, color_buf, depth_buf, width, height = buffers
    glPixelStorei(GL_PACK_ALIGNMENT, 1)
    glReadBuffer(GL_COLOR_ATTACHMENT0)
    data = glReadPixels(0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE)
    return data, width, height

window = None;
buffers = None;
ctx = None;
if pf.platform().startswith('Windows'):
    def runglfw(draw=draw):
        if not glfw.init():
            return;
        global window,buffers;
        if window is None:
            # Set window hint NOT visible
            glfw.window_hint(glfw.VISIBLE, False);
            # Create a windowed mode window and its OpenGL context
            window = glfw.create_window(w,h, "hidden window", None, None);
            glfw.make_context_current(window);
            buffers = myglCreateBuffers(w,h);
        if not window:
            glfw.terminate();
            return;
        initGL();
        reshape(w,h);
        display(draw);
        data, width, height = myglReadColorBuffer(buffers);
        m = np.frombuffer(data,np.uint8);
        m = m.reshape(w,h,4);
        m = np.flip(m,[0,1]);
        return m;
    def doneglfw():
        global window,buffers;
        if not (window is None):
            glfw.destroy_window(window);
            glfw.terminate();
    donegl=doneglfw;
    rungl = runglfw
elif pf.platform().startswith('Linux'):
    def runglmesa(draw=draw):
        global ctx;
        if ctx is None:
            ctx = osmesa.OSMesaCreateContext(OSMESA_RGBA, None);
        buf = arrays.GLubyteArray.zeros((h, w, 4))
        assert(osmesa.OSMesaMakeCurrent(ctx, buf, GL_UNSIGNED_BYTE, w, h));
        assert(osmesa.OSMesaGetCurrentContext())
        initGL();
        reshape(w,h);
        display(draw);
        data = glReadPixels(0, 0, w, h, GL_RGBA, GL_UNSIGNED_BYTE)
        m = np.frombuffer(data,np.uint8);
        m = m.reshape(w,h,4);
        m = np.flip(m,[0,1]);
        return m;
    def doneglmesa():
        global ctx;
        if not (ctx is None):
            osmesa.OSMesaDestroyContext(ctx);
    donegl=doneglmesa;
    rungl = runglmesa
else:
    assert(False,'Unkown platform');

def run(**kwargs):
    data = rungl();
    image = Image.fromarray(data,'RGBA');
    image.save('./log/glut.png');
    return;