from __future__ import print_function
import numpy as np;
from functools import cmp_to_key;
from scipy.spatial.transform import Rotation as R;
from scipy.spatial import Delaunay;
from scipy.spatial import ConvexHull;
from ply import write_ply;
import pandas as pd;
import os;
import json;
from json import JSONEncoder;
box_face = np.array(
[
[1,3,0],
[2,3,1],
[7,5,4],
[7,6,5],
[4,5,0],
[5,1,0],
[1,5,2],
[2,5,6],
[6,3,2],
[6,7,3],
[4,0,3],
[3,7,4]
],
dtype=np.int32
);
box_vert = np.array(
[
#box1
    [
        [-0.3,0.4,0.0],[0.3,0.4,0.0],[0.3,-0.4,0.0],[-0.3,-0.4,0.0],
        [-0.3,0.4,0.2],[0.3,0.4,0.2],[0.3,-0.4,0.2],[-0.3,-0.4,0.2]
    ],
#box2
    [
        [-0.4,0.4,0.0],[0.4,0.4,0.0],[0.4,-0.4,0.0],[-0.4,-0.4,0.0],
        [-0.4,0.4,0.2],[0.4,0.4,0.2],[0.4,-0.4,0.2],[-0.4,-0.4,0.2]
    ],
#box3
    [
        [-0.2,0.2,0.0],[0.2,0.2,0.0],[0.2,-0.2,0.0],[-0.2,-0.2,0.0],
        [-0.2,0.2,0.2],[0.2,0.2,0.2],[0.2,-0.2,0.2],[-0.2,-0.2,0.2]
    ],
#box4
    [
        [-0.1,0.2,0.0],[0.1,0.2,0.0],[0.1,-0.2,0.0],[-0.1,-0.2,0.0],
        [-0.1,0.2,0.2],[0.1,0.2,0.2],[0.1,-0.2,0.2],[-0.1,-0.2,0.2]
    ],
#box5
    [
        [-0.25,0.25,0.0],[0.25,0.25,0.0],[0.25,-0.25,0.0],[-0.25,-0.25,0.0],
        [-0.25,0.25,0.5],[0.25,0.25,0.5],[0.25,-0.25,0.5],[-0.25,-0.25,0.5]
    ]
]
,
dtype=np.float32
);
#box_vert = box_vert*1.0;

box_vol = [0.096,0.128,0.032,0.016,0.125];
box_base = [0.48,0.64,0.16,0.08,0.25];
w = np.linspace(0.0,1.0, num=250);
box_border_w = np.zeros([1000,4,1]);
box_border_w[0:250,0,0] = w;
box_border_w[0:250,1,0] = 1.0 - w;
box_border_w[250:500,1,0] = w;
box_border_w[250:500,2,0] = 1.0 - w;
box_border_w[500:750,2,0] = w;
box_border_w[500:750,3,0] = 1.0 - w;
box_border_w[750:1000,3,0] = w;
box_border_w[750:1000,0,0] = 1.0 - w;