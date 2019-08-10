import mxnet as mx
import numpy as np
from mxnet import autograd, gluon
from mxnet.gluon.data import DataLoader
import gluoncv
from mxnet import nd
import time
from train_csv import Reader

ctx = mx.gpu(2)
batch_size = 64

resnet=gluoncv.model_zoo.get_model('mobilenet1.0',pretrained=False)
resnet=resnet.features
resnet.load_parameters('./8.15_mobilenet_metric_66.params')

class MyNet(gluon.nn.HybridBlock):
    def __init__(self):
        super().__init__()
        self.blk=gluon.nn.HybridSequential()
        self.blk.add(resnet,
                     gluon.nn.Dense(447))

    def hybrid_forward(self, F, x, *args, **kwargs):
        return self.blk(x)


def main():
    net=MyNet()
    net.blk[1:].initialize(init=mx.init.Xavier())
    #pred_box_50.params是Fatten后，直接采用Dense（447）的结果，为4.9。new_pred_box_8.params为4.3的结果
    #net.load_parameters('./pred_box_50.params')
    net.collect_params().reset_ctx(ctx)
    soft_max=gluon.loss.SoftmaxCrossEntropyLoss()
    focal_loss=gluoncv.loss.FocalLoss(num_class=447,size_average=False)
    epoches=101
    transforms = gluon.data.vision.transforms.Compose([

        gluon.data.vision.transforms.RandomSaturation(0.3),
        gluon.data.vision.transforms.RandomContrast(0.3),
        gluon.data.vision.transforms.RandomBrightness(0.1),
        gluon.data.vision.transforms.RandomFlipTopBottom(),
        gluon.data.vision.transforms.RandomResizedCrop((256, 256), (0.9, 1.0)),
        gluon.data.vision.transforms.ToTensor()
    ])
    train_data = Reader('./data/imgs', 'cls',data_argument=True)
    train_data = DataLoader(
        train_data.transform_first(transforms),
        batch_size, True, num_workers=16)
    now_lr = 0.01
    # trainer = gluon.Trainer(net.blk[1:].collect_params(), 'sgd',
    #                         {'learning_rate': now_lr, 'momentum': 0.9})
    metric = mx.metric.Accuracy()
    lr_sch = mx.lr_scheduler.FactorScheduler(step=int(5000 / batch_size * 2), factor=0.9, stop_factor_lr=1e-04,
                                            warmup_steps=int(5000 / batch_size * 20), warmup_begin_lr=1e-02,
                                            warmup_mode='constant')

    trainer = gluon.Trainer(net.blk[-1:].collect_params(), 'sgd',
                            {'learning_rate': 1e-03, 'wd': 1e-06, 'momentum': 0.9, 'lr_scheduler': lr_sch})

    trainer_m = gluon.Trainer(net.blk[:-1].collect_params(), 'sgd', {'learning_rate': 1e-05, 'wd': 0.000001})

    for epoch in range(epoches):
        metric.reset()
        total_loss = 0
        start_time = time.time()
        for i, (data, label) in enumerate(train_data):
            data = data.as_in_context(ctx)
            label = label.as_in_context(ctx).reshape(-1).astype('float32')
            with autograd.record():
                pred=net(data)
                # loss=soft_max(pred,label)
                loss = soft_max(pred, label)
            loss.backward()

            trainer.step(batch_size)
            trainer_m.step(batch_size)
            total_loss = mx.nd.mean(loss).asscalar()
            metric.update(label,pred)
            _,acc=metric.get()
            # if i % 20 == 0:
            #     print('Batch: %s, Loss: %s' % (i, total_loss))

        print('Epoch: %s, Loss: %s, acc: %s,Time: %s' % (epoch, total_loss,acc,time.time()-start_time))
        if epoch>0 and epoch%2==0:
            # net.save_parameters('pred_box_'+str(epoch)+'.params')
            # net.save_parameters('8.10_new_pred_box_' + str(epoch) + '.params')
            net.save_parameters('8.13_crop_pred_box_' + str(epoch) + '.params')

            # #net.save_parameters('together_metric_' + str(epoch) + '.params')
            # is_cluster = False
            # if 'running_package' in sys.argv[0] and 'home' not in sys.argv[0]:
            #     is_cluster = True
            #     project_path = '/running_package'
            #
            # net.save_parameters('together_metric_' + str(epoch) + '.params')
            #
            # if is_cluster:
            #     hdfs_order = 'hdfs dfs -put {}  hdfs://njstorage001.hogpu.cc:8020/user/jiaxin.wang/model'.format(
            #         'together_metric_' + str(epoch) + '.params')
            #     os.system(hdfs_order)


if __name__ == '__main__':
    main()



