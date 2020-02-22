from load_model.mxnet_model_structure import SSD
import mxnet.ndarray as F
from mxnet import  gluon,nd
from mxnet.gluon import nn


def cls_predictor(num_anchors, num_classes, idx):
    blk = nn.Sequential()
    blk.add(nn.Conv2D(64, kernel_size=3, padding=1, prefix= 'cls_%d_insert_conv2d' % idx, activation='relu'))
    blk.add(nn.Conv2D(num_anchors * num_classes, kernel_size=3, padding=1, prefix='cls_%d_conv' %idx))
    return blk


def loc_predictor(num_anchors, num_classes, idx):
    blk = nn.Sequential()
    blk.add(nn.Conv2D(64, kernel_size=3, padding=1, prefix= 'loc_%d_insert_conv2d' % idx, activation='relu'))
    blk.add(nn.Conv2D(num_anchors * 4, kernel_size=3, padding=1, prefix='loc_%d_conv' %idx))
    return blk

def down_sample_blk(num_channels, layer_idx):
    blk = nn.Sequential()
    blk.add(nn.MaxPool2D(2, prefix='maxpool2d_%d' % (layer_idx - 1)))
    blk.add(nn.Conv2D(num_channels, kernel_size=3, padding=1, prefix="conv2d_%d" % layer_idx),
                nn.Activation('relu', prefix= 'conv2d_%d_activation' % layer_idx))
    return blk


class SSD(gluon.Block):
    def __init__(self, **kwargs):
        super(SSD, self).__init__(**kwargs)
        self.filters = [32, 64, 64, 64, 128, 128, 64, 64]
        self.conv_0 = nn.Conv2D(self.filters[0], kernel_size=(3, 3), prefix="conv2d_0", padding=(1, 1), activation='relu')
        for i in range(1, 7):
            setattr(self, 'conv_%d' % i, down_sample_blk(self.filters[i], i))
        self.conv_7 = nn.Conv2D(self.filters[7], kernel_size=(3, 3), prefix="conv2d_7",padding=(0, 0), activation='relu')
        for i in range(5):
            setattr(self, 'cls_%d_conv' % i, cls_predictor(4, 2, i))
            setattr(self, 'loc_%d_conv' %i, loc_predictor(4, 2, i))

    def forward(self, x):
        cls_preds, loc_preds = [None] * 5, [None] * 5
        x = self.conv_0(x)

        for i in range(1, 8):
            x = getattr(self, 'conv_%d' % i)(x)
            if i in [3, 4, 5, 6, 7]:
                cls_x = getattr(self, 'cls_%d_conv' % (i-3))(x)
                cls_preds[i-3] =F.sigmoid(cls_x.transpose((0,2,3,1)).reshape((0,-1,2)))
                loc_x = getattr(self, 'loc_%d_conv' % (i-3))(x)
                loc_preds[i-3]  = loc_x.transpose((0,2,3,1)).reshape((0,-1,4))

            if i in [2,3,4,5]:
                x = F.Pad(x,pad_width=(0,0,0,0,0,1,0,1),mode='edge')
        return nd.concat(*cls_preds, dim=1), nd.concat(*loc_preds, dim=1)


def copy_weight(caffenet, gluonnet):
    gluon_weights = gluonnet.collect_params()
    for key in caffenet.params.keys():
        layer = caffenet.params[key]
        weight = layer[0].data
        bias = layer[1].data
        gluon_weights[key + 'weight'].set_data(weight)
        gluon_weights[key + 'bias'].set_data(bias)
        print('set weights for %s' % key)
        print("caffe weight", weight.sum(), bias.sum())
        print("gluon weight", gluon_weights[key + 'weight'].data().sum(),
              gluon_weights[key + 'bias'].data().sum())


def load_mxnet_model(weight_path):
    ssd = SSD()
    ssd.load_parameters(weight_path)
    return ssd

def mxnet_inference(model, img_arr):
    y_scores, y_bboxes = model.forward(nd.array(img_arr))
    return y_bboxes.asnumpy(), y_scores.asnumpy()