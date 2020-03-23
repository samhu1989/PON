# PON
Part Based Image To Structure

## File Structure
/net    ---network structure  
/config ---configuration files  
/util   ---utility functions  
/scripts ---scripts to run tasks as: train eval debug data-process
/data   ---path or link for data  
/doc    ---project documents  
/docs   ---host project home pages  
/log    ---log file path  
run.py  ---global main function, mainly define the running options

## Dataset

the dataset for training and testing can be downloaded [here](http://171.67.77.236:8082/cagenet.tar.gz)

### Environment Set Up

To run code from this repo, we recommend to use the docker image [samhu/pytorch:latest](https://hub.docker.com/repository/docker/samhu/pytorch/general) from docker hub .

In addition, the chamfer distance used in this project need to be installed 

as 

```bash
pip install chamferdist
```

## Training 

The pipeline in this repo contains three networks that needed to be independently trained.

```
python run.py -X util.trainvalcage -net Touch -config BoxBcd -bs 64 -ds CageNet -dp </path to dataset> -nepoch 400 -lrd 40 -key val -md part -rate 0.3
```





## Result Preview
<iframe src=http://171.67.77.236:8082/bv/data~res~new_Chair_full~_6e1dd008531f95fc707cdefe012d0353_r60>
</iframe>

