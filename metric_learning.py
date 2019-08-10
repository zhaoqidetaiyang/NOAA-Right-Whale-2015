import mxnet as mx
import numpy as np
from mxnet import autograd, gluon
from mxnet.gluon.data import DataLoader
import gluoncv
from mxnet import nd
import time
import random
from train_csv import Reader

ctx = mx.gpu(3)
batch_size = 64

resnet=gluoncv.model_zoo.get_model('mobilenet1.0',pretrained=True,root='./pre_mask')

class Data_argument(gluon.nn.HybridBlock):
    def __init__(self,angle,scale=1.0):
        super().__init__()
        assert type(angle) is list or tuple
        #angle为角度
        self.angle=angle
        self.scale=scale

    def hybrid_forward(self, F, x, *args, **kwargs):
        x=x.asnumpy()
        (h, w) = x.shape[:2]
        theta=random.randint(self.angle[0],self.angle[1])
        rotated=x.rotate(-theta)
        rotated = mx.nd.array(rotated)
        return rotated

def evaluate(net,test_data,ctx):
    feature_true_list=[]
    feature_label_list=[]
    p_feature_list=[]
    p_label_list=[]
    for i,(data,label) in enumerate(test_data):
        data=data.as_in_context(ctx)
        label=label.as_in_context(ctx)
        if i<67:
            feature_true=net(data)
            feature_true_list.append(feature_true)
            feature_label=label
            feature_label_list.append(feature_label)
        elif i>=67 and i<134:
            p_data=data
            p_feature = net(p_data)
            p_feature_list.append(p_feature)
            p_label=label
            p_label_list.append(p_label)
        else:
            break
    feature_true=nd.concatenate(feature_true_list,0)
    feature_label=nd.concatenate(feature_label_list,0)
    p_feature=nd.concatenate(p_feature_list,0)
    p_label=nd.concatenate(p_label_list,0).reshape(-1)
    acc=accuracy_metric(feature_true,feature_label,p_feature,p_label)
    return acc

def accuracy_metric(gallery_features, gallery_label, query_features, query_label):
    B1 = nd.sum(nd.square(gallery_features), axis=1, keepdims=True)
    B2 = nd.sum(nd.square(query_features), axis=1, keepdims=True)
    dist_mat = nd.broadcast_add(B2, B1.T) - 2 * nd.dot(query_features, gallery_features.T)
    label_mask = nd.broadcast_equal(dist_mat, nd.min(dist_mat, axis=1, keepdims=True)).astype('float32')
    pre_label_mat = nd.broadcast_mul(label_mask, gallery_label.reshape(1, -1).astype('float32'))
    pre_label_list = nd.max(pre_label_mat, axis=1)
    cor_num = nd.sum(nd.equal(pre_label_list, query_label.astype('float32')))
    return cor_num.asnumpy()[0] / len(query_label)

def Treplit_hard_loss(net,data,label):
    label = label.reshape(-1, 1)
    label_mat = nd.equal(label, label.T).astype('float32')
    vec = net(data)
    dist_self = nd.sum(nd.square(vec), axis=1, keepdims=True)
    dist_mat = nd.broadcast_add(dist_self, dist_self.T) - 2 * nd.dot(vec, vec.T)
    p_min=nd.log(nd.sum(label_mat*nd.exp(dist_mat),axis=1))
    p_max=nd.log(nd.sum((1-label_mat)*nd.exp(-dist_mat)),axis=1)
    loss=nd.relu(p_min+p_max+1)
    return loss

def contrastive_loss(net,data,label):
    label = label.reshape(-1, 1)
    label_mat = nd.relu(-nd.abs(label - label.T) + 1).astype('float32')
    vec = net(data)
    vec = nd.Flatten(vec)
    dist_self = nd.sum(nd.square(vec), axis=1, keepdims=True)
    dist_mat = nd.broadcast_add(dist_self, dist_self.T) - 2 * nd.dot(vec, vec.T)
    loss = label_mat * dist_mat + nd.relu((1.0 - dist_mat * (1 - label_mat))).astype('float32')
    return loss

def lifted_loss(net,data,label):
    label = label.reshape(-1, 1)
    label_mat = nd.equal(label, label.T).astype('float32')
    vec = net(data)
    dist_self = nd.sum(nd.square(vec), axis=1, keepdims=True)
    dist_mat = nd.broadcast_add(dist_self, dist_self.T) - 2 * nd.dot(vec, vec.T)
    p_row = nd.sum(nd.exp(1.0 - (dist_mat)) * (1 - label_mat), 1, True)
    loss = 1000 * (nd.log(p_row + p_row.T + 1e-5) + dist_mat) * label_mat / (2 * label_mat.sum())
    return loss

def main():

    net=resnet.features
    #net.load_parameters('./new_metric_200.params')
    #net.initialize(init=mx.init.Xavier())
    net.collect_params().reset_ctx(ctx)
    transforms=gluon.data.vision.transforms.Compose([
        gluon.data.vision.transforms.RandomSaturation(0.2),
        gluon.data.vision.transforms.RandomContrast(0.2),
        gluon.data.vision.transforms.RandomBrightness(0.1),
        gluon.data.vision.transforms.RandomFlipTopBottom(),
        gluon.data.vision.transforms.ToTensor()
    ])
    test_dataset = Reader('./data/imgs', 'metric',data_argument=False)
    test_data = gluon.data.DataLoader(test_dataset.transform_first(transforms),
                                      batch_size, True, num_workers=16)
    acc = evaluate(net, test_data, ctx=ctx)
    print('Accuracy: %s' % (acc))
    for epoch in range(201):
        train_data = Reader('./data/imgs', 'metric',data_argument=True)
        train_data = DataLoader(
            train_data.transform_first(transforms),
            batch_size, False,num_workers=16)
        total_loss=0
        start_time=time.time()

        for i,(data,label) in enumerate(train_data):
            data=data.as_in_context(ctx)
            label=label.as_in_context(ctx)

            with autograd.record():
                label=label.reshape(-1,1)
                label_mat = nd.equal(label,label.T).astype('float32')
                vec = net(data)
                vec=nd.Flatten(vec)
                dist_mat = - nd.dot(vec, vec.T)/50
                p_row=nd.sum(nd.exp(1.0-(dist_mat))*(1-label_mat),1,True)
                loss=(nd.log(p_row+p_row.T + 1e-5)+dist_mat)*label_mat
                loss=nd.relu(loss)
            loss.backward()
            now_lr =  0.01*(0.99**epoch)
            trainer_triplet = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': now_lr,'momentum':0.9,'wd':0.00005})
            trainer_triplet.step(batch_size)
            total_loss += mx.nd.mean(loss).asscalar()
            # if i % 20 == 0:
            #     print('Batch: %s, Loss: %s' % (i, total_loss))
        print('Epoch: %s, Loss: %s, Time: %s' % (epoch, total_loss/len(train_data),time.time()-start_time))
        start_time=time.time()
        if epoch>0 and epoch%5==0:
            acc=evaluate(net, test_data, ctx=ctx)
            print('Accuracy: %s,Time: %s'%(acc,time.time()-start_time))
        if epoch>10 and epoch%2==0:
            net.save_parameters('8.15_mobilenet_metric_'+str(epoch)+'.params')


if __name__ == '__main__':
    main()



