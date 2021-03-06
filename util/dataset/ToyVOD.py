from .ToyV import *;
from ..data.gen_c2tbox import *;
from ..data.gen_toybox import center_env;
from scipy.spatial.transform import Rotation as R;
from util.render.rungl import rungl,donegl,calnormal;
from util.data.gen_toybox import box_face;
from functools import partial;
from OpenGL.GL import *;
box_vert = np.array(
    [
        [-0.5,0.5,0.0],[0.5,0.5,0.0],[0.5,-0.5,0.0],[-0.5,-0.5,0.0],
        [-0.5,0.5,1.0],[0.5,0.5,1.0],[0.5,-0.5,1.0],[-0.5,-0.5,1.0]
    ],
    dtype=np.float32
    );
    
def randbox(env):
    #random scale
    s = np.random.uniform(0.1,1.0,[1,3]).astype(np.float32);
    v = box_vert*s;
    #random rotation
    q = np.random.normal(0.0001,1.0,[4]).astype(np.float32);
    norm = np.linalg.norm(q);
    r = R.from_quat(q/norm);
    v = r.apply(v);
    #random translation
    t = np.random.normal(0.0,0.5,[1,3]).astype(np.float32);
    v += t;
    env['idx'].append(np.random.randint(0,100));
    env['box'].append(v.copy());
    env['base'].append(s[0,0]*s[0,1]);
    env['R'].append(r.as_quat());
    env['t'].append(t.reshape(-1));
    return;
    
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
            assert False,'need to choose 2d or 3d in user key for ToyVOD';
        if self.train:
            self.datapath = [None]*(opt['batch_size']*8192);
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
        if not self.train:
            fname = self.datapath[index];
            data = json.load(open(fname,'r'));
        else:
            data = self.gen_on_fly();
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
        if not self.train:
            img = np.array(Image.open(fname.replace('.json','.ply_r_000.png'))).astype(np.float32)/255.0;
            img = img.astype(np.float32);
        else:
            img = rungl(partial(drawenv,env=data,box_face=box_face)).astype(np.float32)/255.0;
        s3d = np.array(data['box'][srcpick]);
        s3d = s3d.astype(np.float32);
        s2d = proj3d(mv(s3d));
        s2d = s2d.astype(np.float32);
        #
        t3d = np.array(data['box'][pick]);
        t3d = t3d.astype(np.float32);
        t2d = proj3d(mv(t3d));
        t2d = t2d.astype(np.float32);
        #
        if self.use_3d:
            pass;
        else:
            s2d[:,2] *= 0.0;
            t2d[:,2] *= 0.0;
            
        c3d = np.concatenate([np.mean(t3d,axis=0,keepdims=True),np.mean(s3d,axis=0,keepdims=True)],axis=0);
        mvc3d = mv(c3d);
        dirc3d = mvc3d[0,:] - mvc3d[1,:];
        coord = car2sph(dirc3d.reshape(1,-1));
        coord = coord.astype(np.float32);
        r = coord[:,0];
        gt = coord[:,1:3];
        gt[:,0] /= np.pi;
        gt[:,1] /= (2*np.pi);
        gt = gt.reshape(2);
        return torch.from_numpy(img.copy()),torch.from_numpy(s2d.copy()),torch.from_numpy(s3d.copy()),torch.from_numpy(t2d.copy()),torch.from_numpy(t3d.copy()),torch.from_numpy(r.copy()),torch.from_numpy(gt.copy()),'boxVOD';
    
    def gen_on_fly(self):
        env={'idx':[],'box':[],'top':[1,1],'base':[],'R':[],'t':[]};
        randbox(env);
        randbox(env);
        norm_env(env);
        return env;
    
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
    val_load = data.DataLoader(val_data,batch_size=16,shuffle=False,num_workers=opt['workers']);
    if not os.path.exists('./log/debug_dataset/'):
        os.mkdir('./log/debug_dataset/');
    print('go over')
    for i, d in enumerate(val_load,0):
        print(i);