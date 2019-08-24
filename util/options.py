import argparse;
parser = argparse.ArgumentParser();

def get_opt():
    global parser;
    #string flags
    parser.add_argument('--execute','-X',type=str,default='None',help='action module name');
    parser.add_argument('--data_path','-data',type=str,default='./data/',help='data path');
    parser.add_argument('--net','-net',type=str,default='AtlasNet',help='network');
    #binary flags:
    parser.add_argument('--ply',action='store_true');
    return parser.parse_args();

def usage():
    global parser;
    parser.print_help();