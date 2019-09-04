import tensorflow as tf;
import os;
import numpy as np;
from functools import partial;
import json;
from PIL import Image
from .ply import write_ply
import pandas as pd;

max_num_part_per_sem = 10
IMAGE_SIZE = 224

def valid_pc(pc):
    if np.mean(pc.max(0) - pc.min(0)) > 1e-9:
        return True;
    else:
        return False;


def parse_record(serialized_example,num_sem,ori_num_ins_per_sem,num_ins_per_sem,trans_table):
    feature_map = {
        'pts': tf.FixedLenFeature([10000, 3], dtype=tf.float32),
        'sem_cnt': tf.FixedLenFeature([num_sem+1], dtype=tf.int64),
        'render_id': tf.FixedLenFeature([1], dtype=tf.int64),
        'model_id': tf.FixedLenFeature([1], dtype=tf.string),
        'anno_id': tf.FixedLenFeature([1], dtype=tf.string),
        'img': tf.FixedLenFeature([224, 224, 3], dtype=tf.int64)
        }
    for j in range(1, num_sem+1):
        feature_map['sem-%03d-pc'%j] = tf.FixedLenFeature([ori_num_ins_per_sem[j], 1000, 3], dtype=tf.float32)
        feature_map['sem-%03d-imgmask'%j] = tf.FixedLenFeature([ori_num_ins_per_sem[j], 224, 224], dtype=tf.int64)
    features = tf.parse_single_example(serialized_example, feature_map)
    query_key = tf.strings.join([features['anno_id'], tf.dtypes.as_string(features['render_id'])], separator='-')
    trans_sparse = tf.strings.split(trans_table.lookup(query_key), sep=';')
    trans_dense = tf.sparse.to_dense(trans_sparse, default_value='')
    trans_mat = tf.reshape(tf.strings.to_number(trans_dense, out_type=tf.float32), [3, 4])
    features['pts'] = tf.matmul(tf.concat([tf.cast(features['pts'], tf.float32), tf.ones((10000, 1), dtype=tf.float32)], 1), trans_mat, transpose_b=True)
    features['sem_cnt'] = tf.minimum(tf.cast(features['sem_cnt'], dtype=tf.int32), max_num_part_per_sem)
    sem_mask_list = [];
    for j in range(1, num_sem+1):
        features['sem-%03d-pc'%j] = features['sem-%03d-pc'%j][:num_ins_per_sem[j]]
        features['sem-%03d-imgmask'%j] = tf.cast(features['sem-%03d-imgmask'%j][:num_ins_per_sem[j]], dtype=tf.float32)
        features['sem-%03d-imgmask'%j] = tf.cast(tf.greater(tf.transpose(tf.image.resize_images(tf.transpose(features['sem-%03d-imgmask'%j], [1, 2, 0]), [IMAGE_SIZE, IMAGE_SIZE]), [2, 0, 1]), 0.5), tf.float32)
        sem_mask_list.append(tf.greater(tf.reduce_sum(features['sem-%03d-imgmask'%j], axis=0, keepdims=True), 0))
        features['sem-%03d-pc'%j] = tf.matmul(tf.concat([features['sem-%03d-pc'%j], tf.ones((num_ins_per_sem[j], 1000, 1), dtype=tf.float32)], 2), tf.tile(tf.expand_dims(trans_mat, 0), [num_ins_per_sem[j], 1, 1]), transpose_b=True)
    bg_mask = tf.logical_not(tf.reduce_any(tf.concat(values=sem_mask_list, axis=0), axis=0))
    sem_mask = tf.transpose(tf.concat(values=[tf.expand_dims(bg_mask, axis=0)] + sem_mask_list, axis=0), [1, 2, 0])
    features['sem_mask'] = tf.cast(sem_mask, dtype=tf.float32)
    features['img'] = tf.cast(features['img'], dtype=tf.float32) / 255
    return features;

def write(data_root,num_sem,mid,rid,aid,img,ins_msk_lst,ins_pc_lst):
    mid = mid.flatten()[0].decode();
    aid = aid.flatten()[0].decode();
    rid = rid[0,0,...]
    print(mid,aid,rid);
    path = os.path.join(data_root,'learn2merge');
    if not os.path.exists(path):
        os.mkdir(path);
    path = os.path.join(path,aid+'_'+str(rid));
    if not os.path.exists(path):
        os.mkdir(path);
    with open(os.path.join(path,'info.txt'), 'w') as f:
        print(mid,file=f);
    img_path = os.path.join(path,'img.png');
    img = img[0,...];
    img = Image.fromarray(np.uint8(img*255.0));
    img.save(img_path);
    for i in range(len(ins_msk_lst)):
        msk = ins_msk_lst[i];
        pc = ins_pc_lst[i];
        msk = msk[0,...];
        pc = pc[0,...];
        for j in range(msk.shape[0]):
            msk_path = os.path.join(path,'msk_%d_%d.png'%(i,j));
            pc_path = os.path.join(path,'pc_%d_%d.ply'%(i,j));
            mskimg = Image.fromarray(np.uint8(msk[j,...]*255.0),'L');
            if valid_pc(pc[j,...]):
                mskimg.save(msk_path);
                write_ply(pc_path,points = pd.DataFrame(pc[j,...]));
    return;

def run(**kwargs):
    data_root = kwargs['data_path'];
    stat_fn = os.path.join(data_root,'./stats/part_count_stats-new/%s-%d-input.txt' % ('Chair', 3));
    num_sem = 0; ori_num_ins_per_sem = [0]; num_ins_per_sem = [0]; sem_name = ['other'];
    with open(stat_fn, 'r') as fin:
        for l in fin.readlines():
            _, x, y, _ = l.rstrip().split()
            num_sem += 1
            sem_name.append(x)
            y = int(y)
            if y > 20:
                y = 20
            ori_num_ins_per_sem.append(y)
            num_ins_per_sem.append(min(y, max_num_part_per_sem))
    num_ins_per_sem = np.array(num_ins_per_sem, dtype=np.int32)
    ori_num_ins_per_sem = np.array(ori_num_ins_per_sem, dtype=np.int32)
    print('NUM SEMANTICS: ', num_sem)
    print('ORI NUM INS PER SEM: ', ori_num_ins_per_sem)
    print('NUM INS PER SEM: ', num_ins_per_sem)
    
    trans_fn = os.path.join(data_root,'./partnet_to_render_viewer_space/%s.json' % 'Chair');
    with open(trans_fn, 'r') as fin:
        trans_dict = json.load(fin)
        
    trans_table = tf.contrib.lookup.HashTable(tf.contrib.lookup.KeyValueTensorInitializer(list(trans_dict.keys()), list(trans_dict.values())), '')
    
    records = [];
    for root, dirs, files in os.walk(data_root):
        for name in files:
            if name.endswith('.tfrecords'):
                records.append(os.path.join(root, name));
    dataset = tf.data.TFRecordDataset(records, compression_type='GZIP', buffer_size=32, num_parallel_reads=4)
    map = partial(parse_record,num_sem=num_sem,ori_num_ins_per_sem=ori_num_ins_per_sem,num_ins_per_sem=num_ins_per_sem,trans_table=trans_table);
    dataset = dataset.map(map,10)
    dataset = dataset.prefetch(32)
    dataset = dataset.batch(1,drop_remainder=True);
    dataset_iterator = dataset.make_initializable_iterator()
    
    handle_pl = tf.placeholder(tf.string, shape=[])
    iterator = tf.data.Iterator.from_string_handle(handle_pl, dataset_iterator.output_types, dataset_iterator.output_shapes)
    
    next_element = iterator.get_next()
    input_img = next_element['img']
    model_id = next_element['model_id']
    render_id = next_element['render_id']
    anno_id = next_element['anno_id']
    gt_sem_cnt = next_element['sem_cnt'][:, 1:]
    gt_sem_mask = next_element['sem_mask']
    
    gt_part_per_sem = dict(); gt_ins_mask_per_sem = dict(); gt_part_pc_per_sem = dict();
    for i in range(1, num_sem+1):
        gt_part_per_sem['sem-%03d'%i] = next_element['sem-%03d-pc'%i]
        gt_ins_mask_per_sem['sem-%03d'%i] = next_element['sem-%03d-imgmask'%i]
        gt_part_pc_per_sem['sem-%03d'%i] = next_element['sem-%03d-pc'%i]
        
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    config.log_device_placement = False
    sess = tf.Session(config=config)
    dataset_handle = sess.run(dataset_iterator.string_handle())
    trans_table.init.run(session=sess)
    sess.run(dataset_iterator.initializer)
    
    while True:
        try:
            ops = [model_id,render_id,anno_id,input_img];
            for i in range(1, num_sem+1):
                ops.append(gt_ins_mask_per_sem['sem-%03d'%i])
                ops.append(gt_part_pc_per_sem['sem-%03d'%i])
            out = sess.run(ops,feed_dict={handle_pl:dataset_handle});
            winfo = [data_root,num_sem]
            out = list(out);
            winfo.extend(out[:4]);
            ins_msk_lst = [];
            ins_pc_lst = [];
            for i in range(4,len(out),2):
                ins_msk_lst.append(out[i]);
                ins_pc_lst.append(out[i+1]);
            winfo.append(ins_msk_lst);
            winfo.append(ins_pc_lst);
            write(*winfo);
        except tf.errors.OutOfRangeError:
            break;
    