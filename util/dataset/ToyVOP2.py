from .ToyV import *;
from ..data.gen_c2tbox import *;
from ..data.gen_toybox import center_env;
from scipy.spatial.transform import Rotation as R;
from util.render.rungl import rungl,donegl,calnormal;
from util.data.gen_toybox import box_face;
from functools import partial;
from OpenGL.GL import *;
from scipy.spatial import ConvexHull,Delaunay;
from scipy.sparse.csgraph import shortest_path;
from PIL import Image;
from scipy import ndimage;
box_vert = np.array(
    [
        [-0.5,0.5,0.0],[0.5,0.5,0.0],[0.5,-0.5,0.0],[-0.5,-0.5,0.0],
        [-0.5,0.5,1.0],[0.5,0.5,1.0],[0.5,-0.5,1.0],[-0.5,-0.5,1.0]
    ],
    dtype=np.float32
    );
    
def randbox2(env):
    s = np.random.uniform(0.1,1.0,[1,3]).astype(np.float32);
    v = box_vert*s;
    #random rotation
    q = np.random.normal(0.0001,1.0,[4]).astype(np.float32);
    norm = np.linalg.norm(q);
    r = R.from_quat(q/norm);
    v = r.apply(v);
    #random translation
    t = -1.0*v[np.random.randint(0,8),:];
    v += t;
    env['idx'].append(np.random.randint(0,100));
    env['box'].append(v.copy());
    env['base'].append(s[0,0]*s[0,1]);
    env['R'].append(r.as_quat());
    env['t'].append(t.reshape(-1));
    
def dist2d(v1,v2):
    return np.sqrt(np.sum((v1-v2)**2));
    
def w2sparse(s):
    return;
    
def im2w(im):
    im = Image.fromarray((im*255.0).astype(np.uint8)).convert('L');
    im = im.resize((112,112));
    im = ( np.array(im) / 255.0 ).astype(np.float32);
    sx = ndimage.sobel(im,axis=0,mode='constant');
    sy = ndimage.sobel(im,axis=1,mode='constant');
    sx = np.array(sx);
    sy = np.array(sy);
    s = np.hypot(sx,sy);
    s -= np.min(np.min(s,axis=0,keepdims=True),axis=1,keepdims=True);
    s = s / np.max(np.max(s,axis=0,keepdims=True),axis=1,keepdims=True);
    return s;

class Data(data.Dataset):
    def __init__(self,opt,train=True):
        self.root = opt['data_path'];
        self.pts_num = opt['pts_num_gt'];
        self.train = train;
        self.datapath = [];
        if '3d' in opt['user_key']:
            self.use_3d = True;
        elif '2d' in opt['user_key']:
            self.use_3d = False;
        else:
            assert False,'need to choose 2d or 3d in user key for ToyVOP';
        if self.train:
            dataroot = os.path.join(self.root,'train');
        else:
            dataroot = os.path.join(self.root,'test');
        fs = os.listdir(dataroot);
        for f in fs:
            if (not 'msk' in f) and f.endswith('.json'):
                self.datapath.append(os.path.join(dataroot,f));
        self.datapath.sort();
                
    def __getitem__(self, index):
        try:
            return self.load(index);
        except Exception as e:
            print(e);
            traceback.print_exc();
            exit();
        
    def load(self,idx):
        index = idx%self.__len__();
        fname = self.datapath[index];
        data = json.load(open(fname,'r'));
        num = len(data['box']);
        if self.train:
            pick = np.random.randint(0,num);
        else:
            pick = idx%num;
        if pick == 0:
            srcpick = np.random.randint(1,num);
        else:
            if num > 2:
                alpha = np.random.randint(0,2)
                srcpick = alpha*np.random.randint(0,bspick)+(1-alpha)*np.random.randint(bspick+1,num);
            else:
                srcpick = 0;
        # 
        s3d = np.array(data['box'][srcpick]);
        cs3d = np.mean(s3d,axis=0);
        s3d = s3d.astype(np.float32);
        s2d = proj3d(mv(s3d));
        s2d = s2d.astype(np.float32);
        #
        t3d = np.array(data['box'][pick]);
        t3d = t3d.astype(np.float32);
        ct3d = np.mean(t3d,axis=0);
        t2d = proj3d(mv(t3d));
        t2d = t2d.astype(np.float32);
        #
        img = np.array(Image.open(fname.replace('.json','.ply_r_000.png'))).astype(np.float32)/255.0;
        img = img.astype(np.float32);
        #
        mins = 1000;
        minsi = -1;
        mint = 1000;
        minti = -1;
        if self.train:
            for i in range(s2d.shape[0]):
                if s2d[i,2] < mins:
                    mins = s2d[i,2];
                    minsi = i;

            for i in range(t2d.shape[0]):
                if t2d[i,2] < mint and dist2d(s2d[minsi,:2],t2d[i,:2]) > 10:
                    mint = t2d[i,2];
                    minti = i;
        else:
            for i in range(s2d.shape[0]):
                if s2d[i,2] < mins and s3d[i,2] < cs3d[2]:
                    mins = s2d[i,2];
                    minsi = i;

            for i in range(t2d.shape[0]):
                if t2d[i,2] < mint and t3d[i,2] < ct3d[2] and dist2d(s2d[minsi,:2],t2d[i,:2]) > 10:
                    mint = t2d[i,2];
                    minti = i;
        #
        f3d = np.zeros([2,3],dtype=np.float32);
        f3d[0,:] = t3d[minti,:];
        f3d[1,:] = s3d[minsi,:];
        mvc3d = mv(f3d);
        dirc3d = mvc3d[0,:] - mvc3d[1,:];
        coord = car2sph(dirc3d.reshape(1,-1));
        coord = coord.astype(np.float32);
        r = coord[:,0];
        gt = coord[:,1:3];
        gt[:,0] /= np.pi;
        gt[:,1] /= (2*np.pi);
        gt = gt.reshape(2);
        return torch.from_numpy(img.copy()),torch.from_numpy(s3d.copy()),torch.from_numpy(t3d.copy()),torch.from_numpy(mvc3d.copy()),torch.from_numpy(r.copy()),torch.from_numpy(gt.copy()),torch.from_numpy(s2d[minsi,:2].copy()),torch.from_numpy(t2d[minti,:2].copy()),'boxVOP2';
    
    def __len__(self):
        return len(self.datapath);
        
    def __del__(self):
        donegl();
        
def run(**kwargs):
    opt = kwargs;
    opt['workers'] = 0;
    opt['pts_num_gt'] = 1200;
    train_data = Data(opt,True);
    val_data = Data(opt,False);
    train_load = data.DataLoader(train_data,batch_size=opt['batch_size'],shuffle=True,num_workers=opt['workers']);
    val_load = data.DataLoader(val_data,batch_size=opt['batch_size'],shuffle=False,num_workers=opt['workers']);
    if not os.path.exists('./log/debug_dataset/'):
        os.mkdir('./log/debug_dataset/');
    print('go over');
    for i, d in enumerate(val_load,0):
        fig = plt.figure();
        fig.add_subplot(121);
        im = d[0].cpu().numpy()[0,...]
        plt.imshow(im);
        s2d = d[6].cpu().numpy()[0,...];
        t2d = d[7].cpu().numpy()[0,...];
        plt.plot(s2d[0],s2d[1],'*');
        plt.plot(t2d[0],t2d[1],'x');
        sobel = im2w(im);
        fig.add_subplot(122);
        plt.imshow(sobel,cmap=plt.cm.gray);
        plt.show();
        