import os;
sstk_path = '/home/sstk/ssc';
def render(inputobj,config):
    cwd = os.getcwd();
    os.chdir(sstk_path);
    os.system('node ./render-file.js --input %s --config_file %s --output_dir %s'%(inputobj,config,os.path.dirname(inputobj)));
    os.chdir(cwd);
    