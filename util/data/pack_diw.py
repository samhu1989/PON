import h5py;
import os;
from PIL import Image;
import matplotlib.pyplot as plt;
import math;
def run(**kwargs):
    droot = kwargs['data_path'];
    fname = os.path.join(droot,'DIW_Annotations','DIW_train_val.csv');
    with open(fname,'r') as fp:
        for line in fp:
            if line.startswith('./'):
                l = line.rstrip('\n');
                path = os.path.join(droot,'DIW_train_val',os.path.basename(l));
                img = Image.open(path);
                gtline = fp.readline();
                gtnum = gtline.split(',');
                #
                g_input_width = img.size[0];
                g_input_height = img.size[1];
                #
                orig_img_width  = float(gtnum[5])
                orig_img_height = float(gtnum[6])
                #
                y_A_float_orig = float(gtnum[0]) / orig_img_height 
                x_A_float_orig = float(gtnum[1]) / orig_img_width
                y_B_float_orig = float(gtnum[2]) / orig_img_height
                x_B_float_orig = float(gtnum[3]) / orig_img_width
                #
                y_A = min(g_input_height, max(1,math.floor( y_A_float_orig * g_input_height )))
                x_A = min(g_input_width,max(1,math.floor( x_A_float_orig * g_input_width)))
                y_B = min(g_input_height,max(1,math.floor( y_B_float_orig * g_input_height)))
                x_B = min(g_input_width,max(1,math.floor( x_B_float_orig * g_input_width)))
                plt.imshow(img);
                if gtnum[4] == '>':
                    plt.plot(x_A-1,y_A-1,color='m',marker='x',markersize=12)
                    plt.plot(x_B-1,y_B-1,color='b',marker='x',markersize=12)
                elif gtnum[4] == '<':
                    plt.plot(x_A-1,y_A-1,color='b',marker='x',markersize=12)
                    plt.plot(x_B-1,y_B-1,color='m',marker='x',markersize=12)
                plt.show();
    return;
