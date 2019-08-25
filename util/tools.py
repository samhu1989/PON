import os
import random
import numpy as np

class AverageValueMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0.0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
        
class AvgMeterGroup(object):
    def __init__(self,name):
        self.name = name;
        self.overall_meter = AverageValueMeter();
        self.category_meters = {};
        
    def reset(self):
        self.overall_meter.reset();
        for v in self.category_meters.values():
            v.reset();
            
    def update(self,val,cat):
        if isinstance(val,torch.Tensor):
            val = val.cpu().numpy();
        for i,c in enumerate(cat):
            self.overall_meter.update(val[i]);
            if c in self.catgory_meters.keys():
                self.category_meters[c].update(val[i]);
            else:
                self.category_meters[c] = AverageValueMeter();
                self.category_meters[c].update(val[i]);
                
    def __str__(self):
        ret = self.name;
        ret += ',mean:%10.6f'%self.overall_meter.avg;
        for k,v in self.category_meters.items():
            ret += ','+k+':%10.6f'%v.avg;
        return ret;