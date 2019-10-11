import torch;
from torch import Tensor;
import torch.nn as nn;
import traceback
import importlib
from torch.utils.data import DataLoader;
from torch import optim;
from .tools import *;
torch.backends.cudnn.enabled = False;

def data2cuda(data):
    for i in range(len(data)):
        if torch.cuda.is_available() and isinstance(data[i],torch.Tensor):
            data[i] = data[i].cuda();
            data[i].requires_grad = True;
            
def compute_gradient_penalty(D,img, real_samples, fake_samples):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = torch.Tensor(np.random.random((real_samples.size(0), 1, 1)));
    alpha = alpha.type(real_samples.type());
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1.0 - alpha) * fake_samples)).requires_grad_(True);
    d_interpolates = D([img,interpolates]);
    fake = torch.ones([real_samples.shape[0],1],requires_grad=False);
    fake = fake.type(real_samples.type());
    # Get gradient w.r.t. interpolates
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty;

def run(**kwargs):
    #get configuration
    try:
        config = importlib.import_module('config.'+kwargs['config']);
        opt = config.__dict__;
        for k in kwargs.keys():
            if not kwargs[k] is None:
                opt[k] = kwargs[k];
    except Exception as e:
        print(e);
        traceback.print_exc();
        exit();
    #get network
    try:
        names = opt['net'].split('+');
        gname = names[0];
        dname = names[1];
        gm = importlib.import_module('net.g.'+gname);
        dm = importlib.import_module('net.d.'+dname);
        mlst = [gm.Net(**opt),dm.Net(**opt)];
        net = nn.ModuleList(mlst);
        if torch.cuda.is_available():
            net = net.cuda();
        gnet = net[0];
        dnet = net[1];
    except Exception as e:
        print(e);
        traceback.print_exc();
        exit();
    #get dataset
    try:
        m = importlib.import_module('util.dataset.'+opt['dataset']);
        train_data = m.Data(opt,True);
        val_data = m.Data(opt,False);
        train_load = DataLoader(train_data,batch_size=opt['batch_size'],shuffle=True,num_workers=opt['workers']);
        val_load = DataLoader(val_data,batch_size=opt['batch_size'],shuffle=False,num_workers=opt['workers']);
    except Exception as e:
        print(e);
        traceback.print_exc();
        exit();
    
    #run the code
    optim_g = eval('optim.'+opt['optim'])(gnet.parameters(), lr=opt['lr'], betas=(0.5, 0.999));
    optim_d = eval('optim.'+opt['optim'])(dnet.parameters(), lr=opt['lr'], betas=(0.5, 0.999));
    sched_g = torch.optim.lr_scheduler.StepLR(optim_g, step_size=1000, gamma=0.9);
    sched_d = torch.optim.lr_scheduler.StepLR(optim_d, step_size=1000, gamma=0.9);
    
    #load pre-trained
    if opt['model']!='':
        partial_restore(net,opt['model']);
        print("Previous weights loaded");
    
    for iepoch in range(opt['nepoch']):
        if iepoch % 10 == 0:
            sched_g.step();
            sched_d.step();
        if iepoch % opt['print_epoch'] == 0:
            net.eval();
            for i, data in enumerate(val_load,0):
                val_meters = {};
                with torch.no_grad():
                    data2cuda(data);
                    img = data[0];
                    real_pc = data[2];
                    z = torch.randn([img.size(0),opt['latent_dim']],requires_grad=True);
                    z = z.type(img.type());
                    gout = gnet([img,z]);
                    fake_validity = dnet([img,gout['pts']]);
                    real_validity = dnet([img,real_pc]);
                    dout = [real_validity,fake_validity]
                    acc = config.accuracy(data,gout,dout);
                    for k,v in acc.items():
                        if k in val_meters.keys():
                            val_meters[k].update(v,data[-1]);
                        else:
                            val_meters[k] = AvgMeterGroup(k);
                            val_meters[k].update(v,data[-1]);
                config.writelog(net=net,data=data,out=[gout,dout],meter=val_meters,opt=opt,iepoch=iepoch,idata=i,ndata=len(val_data),istraining=False);
        net.train();
        train_meters = {};
        for i, data in enumerate(train_load,0):
            optim_d.zero_grad();
            data2cuda(data);
            img = data[0];
            real_pc = data[2];
            z = torch.randn([img.size(0),opt['latent_dim']],requires_grad=True);
            z = z.type(img.type());
            with torch.no_grad():
                gout = gnet([img,z]);
                gen_pc = gout['pts'];
            real_validity = dnet([img,real_pc]);
            fake_validity = dnet([img,gen_pc.detach()]);
            #
            w_loss = torch.mean(fake_validity) - torch.mean(real_validity);
            gradient_penalty = opt['lambda_gp'] * compute_gradient_penalty(dnet,img,real_pc.data,gen_pc.data);
            d_loss = gradient_penalty - w_loss;
            d_loss.backward();
            optim_d.step()
            if i % opt['train_g'] == 0:
                optim_g.zero_grad();
                z = torch.randn([img.size(0),opt['latent_dim']],requires_grad=True);
                z = z.type(img.type());
                gout = gnet([img,z]);
                fake_validity = dnet([img,gout['pts']]);
                g_loss = - torch.mean(fake_validity)
                g_loss.backward()
                optim_g.step();
            dout = [real_validity,fake_validity];
            acc = config.accuracy(data,gout,dout);
            for k,v in acc.items():
                if k == 'overall':
                    continue;
                if k in train_meters.keys():
                    train_meters[k].update(v,data[-1]);
                else:
                    train_meters[k] = AvgMeterGroup(k);
                    train_meters[k].update(v,data[-1]);
            config.writelog(net=net,data=data,out=[gout,dout],meter=train_meters,opt=opt,iepoch=iepoch,idata=i,ndata=len(train_data),istraining=True);