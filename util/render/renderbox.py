import numpy as np;
from OpenGL.GL import *;

def init(mv,proj):
    glLoadIdentity();
    glMultMatrix(invCamTrafo);
    glMultMatrix(modelTrafo);
    return;
    
def render():
    return;