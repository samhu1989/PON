from .ToyV import *;
from ..data.gen_c2tbox import *;
from ..data.gen_toybox import center_env;
from scipy.spatial.transform import Rotation as R;
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
    env['box'].append(v);
    env['base'].append(s[0,0]*s[0,1]);
    env['R'].append(r.as_quat());
    env['t'].append(t.reshape(-1));
    return;

class Data(data.Dataset):
    def __init__(self,opt,train=True):
        self.opt = opt;
        self.root = opt['data_path'];
        self.pts_num = opt['pts_num_gt'];
        self.train = train;
        self.datapath = [];
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
        if not self.train:
            img = np.array(Image.open(fname.replace('.json','.ply_r_000.png'))).astype(np.float32)/255.0;
            img = img.astype(np.float32);
        else:
            img = np.ones([224,224,3],dtype=np.float32);
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
        if self.opt['user_key'] == '3d':
            pass;
        elif self.opt['user_key'] == '2d':
            s2d[:,2] *= 0.0;
            t2d[:,2] *= 0.0;
        else:
            assert False,'need to choose 2d or 3d in user key for ToyVOD';
            
        c3d = np.concatenate([np.mean(t3d,axis=0,keepdims=True),np.mean(s3d,axis=0,keepdims=True)],axis=0);
        mvc3d = mv(c3d);
        dirc3d = mvc3d[0,:] - mvc3d[1,:];
        coord = car2sph(dirc3d.reshape(1,-1));
        coord = coord.astype(np.float32);
        #s3d = self.mv(s3d);
        #t3d = self.mv(t3d);
        r = coord[:,0];
        gt = coord[:,1:3];
        gt[:,0] /= np.pi;
        gt[:,1] /= (2*np.pi);
        gt = gt.reshape(2);
        return torch.from_numpy(img),torch.from_numpy(s2d),torch.from_numpy(s3d),torch.from_numpy(t2d),torch.from_numpy(t3d),torch.from_numpy(r),torch.from_numpy(gt),'boxVOD';
    
    def gen_on_fly(self):
        env={'idx':[],'box':[],'top':[1,1],'base':[],'R':[],'t':[]};
        randbox(env);
        randbox(env);
        norm_env(env);
        return env;
    
    def __len__(self):
        return len(self.datapath);
        
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