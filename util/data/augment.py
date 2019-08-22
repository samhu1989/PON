import numpy as np;
import trimesh;
from .deal_partnet import seen_cat,unseen_cat,cat_size;
import random;
import os;
import json;

def contact(m1,m2):
    if m1.scale < m2.scale:
       return (m1.convex_hull.contains(m2.vertices)).any();
    else:
       return (m2.convex_hull.contains(m1.vertices)).any();

def augment(mesh_lst,opath):
    for i,mesh_path in enumerate(mesh_lst):
        mesh = trimesh.load(mesh_path);
        if i==0:
            omesh = mesh;
            continue;
        cnt = 0;
        while not contact(mesh,omesh):
            t = omesh.centroid - mesh.centroid;
            norm = np.linalg.norm(t);
            s = norm - mesh.scale - omesh.scale;
            if s > 0:
                T = trimesh.transformations.translation_matrix(t / norm * s);
            else:
                T = trimesh.transformations.translation_matrix(t / norm * (mesh.scale + omesh.scale) * 0.01 );
            mesh.apply_transform(T);
            cnt += 1;
            if cnt > 1000:
                break;
        omesh += mesh;
    omesh.export(opath+os.sep+'objs.ply');
    file = open(opath+os.sep+'objs.json','w',encoding='utf-8')
    json.dump(mesh_lst,file);
    return;
    
def run(**kwargs):
    data_root = kwargs['data_path'];
    seen_list = [];
    for cat in seen_cat:
        cat_root = data_root + os.sep + cat;
        flst = os.listdir(cat_root);
        for f in flst:
            if os.path.isdir(cat_root+os.sep+f):
                seen_list.append(cat_root+os.sep+f);
    random.shuffle(seen_list);
    nofill = open('./nofilled.txt','w');
    print("#nonfilled",file=nofill);
    for i in range(len(seen_list)):
        print(i,'/',len(seen_list));
        part_lst = [];
        p1 = seen_list[i]+os.sep+'objs';
        if i == len(seen_list):
            p2 = seen_list[i+1]+os.sep+'objs';
        else:
            p2 = seen_list[0]+os.sep+'objs';
        fp1 = os.listdir(p1);
        fp2 = os.listdir(p2);
        for f in fp1:
            part_lst.append(p1+os.sep+f);
        for f in fp2:
            part_lst.append(p2+os.sep+f);
        opath = data_root+os.sep+'augment'+os.sep+'%04d'%i;
        if not os.path.exists(opath):
            os.mkdir(opath);
        if os.path.exists(opath+os.sep+'objs.ply'):
            continue;
        random.shuffle(part_lst);
        try:
            if len(part_lst) > 8:
                augment(part_lst[:8],opath);
            else:
                augment(part_lst,opath);
        except:
            print('%04d'%i,file=nofill);
            continue;
        
        
        
    