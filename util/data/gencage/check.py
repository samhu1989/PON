import h5py;
import numpy as np;

if __name__ == '__main__':
    f = h5py.File('E:\WorkSpace\PONData\check.h5','r');
    print(f.keys());
    print(np.array(f['box']).shape);
    print(np.array(f['img448']).shape);
    print(np.array(f['touch']).shape);
    print(np.max(np.array(f['touch'])));
    print(np.array(f['touch']));
    