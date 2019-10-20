import argparse;
parser = argparse.ArgumentParser();

def get_opt():
    global parser;
    #string flags
    parser.add_argument('--execute','-X',type=str,default='None',help='action module name');
    parser.add_argument('--data_path','-dp',type=str,default='./data/',help='data path');
    parser.add_argument('--dataset','-ds',type=str,default='PON',help='dataset loader');
    parser.add_argument('--net','-net',type=str,default='AtlasNet',help='network');
    parser.add_argument('--model','-mp',type=str,default='',help='pre-trained');
    parser.add_argument('--config','-config',type=str,default='AtlasConfig',help='network configuration');
    parser.add_argument('--log','-log',type=str,default='./log',help='log path');
    parser.add_argument('--user_key','-key',type=str,default='',help='costumized key string');
    parser.add_argument('--input','-i',type=str,default='',help='input filename');
    parser.add_argument('--output','-o',type=str,default='',help='output filename');
    #binary flags:
    parser.add_argument('--ply',action='store_true');
    #int flags
    parser.add_argument('--batch_size','-bs',type=int,default=32,help='batch size');
    parser.add_argument('--nepoch','-nepoch',type=int,help='epoch number');
    parser.add_argument('--lr_decay_freq','-lrd',type=int,default=1,help='decay lr every few epochs');
    #int flags
    parser.add_argument('--lr','-lr',type=float,help='learning rate');
    return parser.parse_args();

def usage():
    global parser;
    parser.print_help();