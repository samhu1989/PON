from util.render.rungl import rungl,donegl;
from util.dataset.ToyVOD import *;
from .gen_toybox import box_face;
from PIL import Image;
import json;
from json import JSONEncoder;
import matplotlib.pyplot as plt;
import os;

def gen_on_fly():
    env={'idx':[],'box':[],'top':[1,1],'base':[],'R':[],'t':[]};
    randbox2(env);
    randbox2(env);
    norm_env(env);
    return env;
    
def drawenv(env,box_face):
    glBegin(GL_TRIANGLES);
    for bi in range(len(env['box'])):
        for i in range(box_face.shape[0]):
            p = env['box'][bi][box_face[i,:],:];
            n = calnormal(p[0,:].copy(),p[1,:].copy(),p[2,:].copy());
            glNormal3f(n[0],n[1],n[2]);
            for j in range(p.shape[0]):
                glVertex3f(p[j,0],p[j,1],p[j,2]);
    glEnd();
    glFlush();
    
def drawenvb(env,box_face,bi):
    glBegin(GL_TRIANGLES);
    for i in range(box_face.shape[0]):
        p = env['box'][bi][box_face[i,:],:];
        n = calnormal(p[0,:].copy(),p[1,:].copy(),p[2,:].copy());
        glNormal3f(n[0],n[1],n[2]);
        for j in range(p.shape[0]):
            glVertex3f(p[j,0],p[j,1],p[j,2]);
    glEnd();
    glFlush();
    
def run(**kwargs):
    root = "./data/hs/"
    if kwargs['user_key'] == 'debug':
        debug = True;
    print(kwargs["user_key"]);
    for i in range(128):
        path = os.path.join(root,'%03d'%i);
        print(i);
        if os.path.exists(path):
            continue;
        else:
            os.mkdir(path);
        env = gen_on_fly();
        img = rungl(partial(drawenv,env=env.copy(),box_face=box_face));
        if debug:
            plt.subplot(311)
            plt.imshow(img);            
        query = os.path.join(path,'query.png');
        Image.fromarray(img).save(query);
        img = rungl(partial(drawenvb,env=env.copy(),box_face=box_face,bi=0));
        if debug:
            plt.subplot(312)
            plt.imshow(img);
        A = os.path.join(path,'A.png');
        Image.fromarray(img).save(A);
        img = rungl(partial(drawenvb,env=env.copy(),box_face=box_face,bi=1));
        if debug:
            plt.subplot(313)
            plt.imshow(img);
        B = os.path.join(path,'B.png');
        Image.fromarray(img).save(B);
        info = os.path.join(path,'info.json');
        json.dump(env,open(info,'w'),cls=Encoder);
        if debug:
            plt.show();

    