import os
import pandas as pd
import mxnet as mx
from datetime import datetime
import pathlib
import shutil
import subprocess
import cv2
from mxnet import gluon, nd, image
from mxnet.gluon.loss import Loss, _apply_weighting, _reshape_like
import numpy as np
from varname.helpers import debug

def listfind(set_, subset_):
    '''
    returns found (ids_of_subset_found_objects_in_set, found_values)
    '''
    return (
        [j for j in range(len(set_)) if set_[j] in subset_],
        [x for x in set_ if x in subset_]
    )
    

def list2nonEmptyIds(l):
    return [j for j in range(len(l)) if l[j]]

def roundUp(a):
    '''
    округление всегда в большую сторону
    '''
    b = round(a)
    if b < a:
        b += 1
    return b

def is_number(s):
    try:
        float(s)
        return True
    except:
        return False

def denorm_shifted_log(data):
    if is_number(data):
        if data <= 0:
            return data
        return np.exp(float(data))-1
    raise ValueError("norm_shifted_log::not a number {}".format(data))

def norm_shifted_log(data):
    if is_number(data):
        if data <= 0:
            return data
        return np.log(float(data) + 1)
    raise ValueError("norm_shifted_log::not a number {}".format(data))

class SoftmaxOutputCrossEntropyLoss(Loss):
    r"""Computes the softmax cross entropy loss for network with softmax out
    """

    def __init__(self, axis=-1, sparse_label=True, from_logits=False, weight=None,
                 batch_axis=0, **kwargs):
        super(SoftmaxOutputCrossEntropyLoss, self).__init__(
            weight, batch_axis, **kwargs)
        self._axis = axis
        self._sparse_label = sparse_label
        self._from_logits = from_logits

    def hybrid_forward(self, F, pred, label, sample_weight=None):
        if not self._from_logits:
            pred = F.log(pred)
        if self._sparse_label:
            loss = -F.pick(pred, label, axis=self._axis, keepdims=True)
        else:
            label = _reshape_like(F, label, pred)
            loss = -F.sum(pred * label, axis=self._axis, keepdims=True)
        loss = _apply_weighting(F, loss, self._weight, sample_weight)
        return F.mean(loss, axis=self._batch_axis, exclude=True)

class SingleClassAccuracy(mx.metric.EvalMetric):
    def __init__(self, name='single_class_accuracy',
                 output_names=None, label_names=None):
        super(SingleClassAccuracy, self).__init__(
            name,
            output_names=output_names, 
            label_names=label_names
        )

    def update(self, label, preds):
        pred_label = nd.argmax(preds, axis=0)
        pred_label = pred_label.asnumpy().astype('int32')
        if pred_label[0] == label:
            self.sum_metric += 1
        self.num_inst += 1

def ls(src, sort=True):
    """os.listdir modify with check os.path.isdir(src)
    
    Arguments:
        src {[str]} -- [dir path]
    
    Returns:
        [False] -- [if src is not dir]
    """
    if os.path.isdir(src):
        if sort:
            return sorted(os.listdir(src))
        else:
            return os.listdir(src)
    raise NotADirectoryError('not a dir', src)

def ls_wc(src):
    return sum([len(files) for r, d, files in os.walk(src)])

def rmdir(dir_p):
    shutil.rmtree(dir_p)

def rmrf(path):
    if os.path.isdir(path):
        shutil.rmtree(path)
    os.remove(path)

def split_path(path):
    return os.path.normpath(path).split(os.path.sep)

def isiter(obj):
    try:
        _ = (e for e in obj)
    except TypeError:
        return False
    return True

def flat_list(l=[]):
    return [item for sublist in l for item in sublist]

def find_files(dir, search_substr):
    found_files = []
    for obj_name in ls(dir):
        obj_path = os.path.join(dir, obj_name)
        if os.path.isdir(obj_path):
            found_files += find_files(obj_path, search_substr)
        if search_substr in obj_name:
            found_files.append(obj_path)
    return found_files
    
def cp_r(src, dst):
    ensure_folder(os.path.split(dst)[0])
    if os.path.isfile(src):
        shutil.copy(src, dst)
        return True
    if os.path.isdir(src):
        for item in ls(src):
            item_p = os.path.join(src, item)
            if os.path.isfile(item_p):
                shutil.copy(item_p, dst)
                return True
            elif os.path.isdir(item_p):
                new_dst = os.path.join(dst, item)
                ensure_folder(new_dst)
                cp_r(item_p, new_dst)
    return False


def acc_on_data(net, val_data, ctx, is_binary=False):
    if is_binary:
        metric = mx.metric.F1()
    else:
        metric = mx.metric.Accuracy()
    for i, batch in enumerate(val_data):
        data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0, even_split=False)
        label = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0, even_split=False)
        outputs = [net(X) for X in data]
        metric.update(label, outputs)
    return metric.get()

def acc_on_single_class(net, val_data, ctx, classes_number):
    metrics = []
    for i in range(classes_number):
        metrics.append(SingleClassAccuracy())
    for i, batch in enumerate(val_data):
        data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0, even_split=False)
        label = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0, even_split=False)
        outputs = [net(X) for X in data]
        for label_ind in range(len(label[0])):
            for i in range(classes_number):
                if label[0][label_ind] == i:
                    metrics[i].update(label[0][label_ind].asnumpy()[0], outputs[0][label_ind])
    metrics_answ = []
    for m in metrics:
        metrics_answ.append(m.get())
    return metrics_answ


def ensure_folder(dir_fname):
    if not os.path.exists(dir_fname):
        try:
            pathlib.Path(dir_fname).mkdir(parents=True, exist_ok=True)
        except PermissionError:
            print('Unable to create {} directory. Permission denied'.format(dir_fname))
            return False
    return True

def str2bool(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string, use False or True')
    return s == 'True'

def show_img(wind_name, img, is_bgr=False):
    img2show = img
    if isinstance(img, mx.ndarray.ndarray.NDArray):
        img2show = img2show.asnumpy()
    if not is_bgr:
        img2show = cv2.cvtColor(img2show, cv2.COLOR_RGB2BGR)
    cv2.imshow(wind_name, img2show)
    cv2.waitKey(0)

def img2batch(img):
    return img.expand_dims(axis=0)

def imgs2batches(imgs, batch_size, shuffle=False):
    return mx.io.NDArrayIter(data=nd.concatenate(imgs), 
                             batch_size=batch_size, shuffle=shuffle)

def load_img(fname, gpu=True, is_gsc=False, to_tensor=False):
    flag = 1
    if is_gsc:
        flag = 0 
    img = mx.image.imread(fname, flag=flag)
    if gpu:
        img = img.copyto(mx.gpu())
    
    if to_tensor:
        return mx.nd.image.to_tensor(img)
    return img

def read_img(fname, is_gsc=False):
    if is_gsc:
        return mx.image.imread(fname,0)
    return mx.image.imread(fname)

def transform2size(img, inp_size, toTensor=False):
    if toTensor:
        transform = transforms.Compose([
            transforms.Resize(inp_size),
            transforms.ToTensor()
        ])
        return transform(img)
    transform = transforms.Compose([
            transforms.Resize(inp_size)
        ])
    return transform(img)
    
def tensor2img(tens):
    return nd.transpose(tens, [1,2,0])

def shell(command_sh):
    try:
        process = subprocess.Popen(command_sh.split(), stdout=subprocess.PIPE)
    except Exception as e:
        print('shell:: subprocess.Popen', command_sh, e)
    try:
        output, error = process.communicate()
        return output, error
    except Exception as e:
        print('shell:: subprocess.communicate()', command_sh, e)
    return None

def curDateTime():
    now = datetime.now()
    return now.strftime("%d.%m.%Y::%H:%M:%S")

class Softmax(mx.gluon.nn.HybridBlock):
    def __init__(self, **kwargs):
         super(Softmax, self).__init__(**kwargs)
         
    def hybrid_forward(self, F, x):        
        return F.softmax(x)

def readSearchXlsxReport(file_path, sheet_name=''):
    file_abs_path = os.path.abspath(file_path)
    print('readSearchGUIReport::reading data at::', file_abs_path)
    df_sheet_all = pd.read_excel(file_path, sheet_name=None, engine='openpyxl')
    if len(sheet_name) != 0:
        try:
            return df_sheet_all[sheet_name]
        except KeyError as e:
            print(e)
            print('readSearchGUIReport::sheets in file::')
            [print(x) for x in df_sheet_all]
            raise Exception(
                'readSearchGUIReport::selected sheet {} is not accessible for file {}'.format(
                    sheet_name, file_path
                )
            )
    selected_sheet_name = list(df_sheet_all.keys())[0]
    if len(df_sheet_all) > 1:
        selected_sheet_name = [x for x in df_sheet_all if 'Total_Report' in x]
        if len(selected_sheet_name) == 0:
            raise Exception('more than one sheet and no one with Total_Report')
        selected_sheet_name = selected_sheet_name[0]
    print('readSearchGUIReport::selected_sheet_name::', selected_sheet_name)
    return df_sheet_all[selected_sheet_name]