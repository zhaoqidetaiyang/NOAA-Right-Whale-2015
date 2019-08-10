import mxnet as mx
from mxnet import gluon,autograd
import gluoncv
from train_csv import Reader
import pandas as pd
import time
from mxnet.gluon import nn

resnet=gluoncv.model_zoo.get_model('resnet18_v2',pretrained=True,root='pre_mask')
class MyNet(gluon.nn.HybridBlock):

    def __init__(self):
        super().__init__()
        self.blk=gluon.nn.HybridSequential()
        self.blk.add(resnet.features,
                     gluon.nn.Dense(256,activation='relu'),
                     gluon.nn.Dense(2))

    def hybrid_forward(self, F, x, *args, **kwargs):
        out=self.blk(x)
        return out
#
# class loss_hard(nn.HybridBlock):
#     def __init__(self,batch,factor):
#         super().__init__()
#         self.l2=gluon.loss.L2Loss()
#         self.batch=batch
#         self.factor=factor
#
#     def hybrid_forward(self, F, pred,label ,*args, **kwargs):
#         label=label.astype('float32')
#         l=self.l2(pred,label)
#         l_1=mx.nd.sort(l)
#         id=int(self.factor*self.batch)
#         thresh=l_1[id]
#         return mx.nd.maximum(l,thresh)

net=MyNet()
net.blk[1:].initialize(init=mx.init.Xavier())
#net.load_parameters('./theta_regression_tanh_9.params')
batch_size=8
epochs=30
ctx=mx.cpu()
net.collect_params().reset_ctx(ctx)

transforms=gluon.data.vision.transforms.Compose(
    [gluon.data.vision.transforms.RandomBrightness(0.3),
     gluon.data.vision.transforms.RandomSaturation(0.3),
     gluon.data.vision.transforms.RandomContrast(0.3),
     gluon.data.vision.transforms.ToTensor(),
     gluon.data.vision.transforms.Normalize(0,1)])
# data=pd.read_csv('./data/train_with_annotations_vinh.csv')
# t_data=data.values
train_data=Reader('./data/imgs','theta')
train_loader=gluon.data.DataLoader(train_data.transform_first(transforms),
                                   batch_size=batch_size,shuffle=True,num_workers=16)
metric=mx.metric.MAE()
trainer = gluon.Trainer(
    net.collect_params(), 'sgd',
    {'learning_rate': 0.001,'momentum':0.9,'wd':0.0005})
# loss=gluon.loss.L1Loss()
loss=gluon.loss.L1Loss()
for epoch in range(epochs):
    total_loss=0
    tic = time.time()
    btic = time.time()
    metric.reset()
    for i,(data,label) in enumerate(train_loader):
        data=data.as_in_context(ctx)
        label=label.as_in_context(ctx).astype('float32')
        with autograd.record():
            output=net(data)
            l=loss(output,label)
        l.backward()

        trainer.step(batch_size)
        total_loss += l.mean().asscalar()
        metric.update(label,output)
        _,acc=metric.get()

        if i % 100 == 0:
            print('[Epoch {}][Batch {}], Speed: {:.3f} samples/sec, loss={:.5f}'.format(
                epoch, i, batch_size / (time.time() - btic), total_loss/(i+1)))
    net.save_parameters('./theta_regression_tanh_'+str(epoch)+'.params')

