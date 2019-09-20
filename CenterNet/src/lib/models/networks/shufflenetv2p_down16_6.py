import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from collections import OrderedDict
from torch.nn import init
import math
BN_MOMENTUM = 0.1
def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )



class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()

    channels_per_group = num_channels // groups
    
    # reshape
    x = x.view(batchsize, groups, 
        channels_per_group, height, width)

    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width)

    return x

def fill_fc_weights(layers):
    for m in layers.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, std=0.001)
            # torch.nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
            # torch.nn.init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
                
class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, benchmodel):
        super(InvertedResidual, self).__init__()
        self.benchmodel = benchmodel
        self.stride = stride
        assert stride in [1, 2]

        oup_inc = oup//2
        
        if self.benchmodel == 1:
            #assert inp == oup_inc
        	self.banch2 = nn.Sequential(
                # pw
                nn.Conv2d(oup_inc, oup_inc, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup_inc),
                nn.ReLU(inplace=True),
                # dw
                nn.Conv2d(oup_inc, oup_inc, 3, stride, 1, groups=oup_inc, bias=False),
                nn.BatchNorm2d(oup_inc),
                # pw-linear
                nn.Conv2d(oup_inc, oup_inc, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup_inc),
                nn.ReLU(inplace=True),
            )                
        else:                  
            self.banch1 = nn.Sequential(
                # dw
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                # pw-linear
                nn.Conv2d(inp, oup_inc, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup_inc),
                nn.ReLU(inplace=True),
            )        
    
            self.banch2 = nn.Sequential(
                # pw
                nn.Conv2d(inp, oup_inc, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup_inc),
                nn.ReLU(inplace=True),
                # dw
                nn.Conv2d(oup_inc, oup_inc, 3, stride, 1, groups=oup_inc, bias=False),
                nn.BatchNorm2d(oup_inc),
                # pw-linear
                nn.Conv2d(oup_inc, oup_inc, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup_inc),
                nn.ReLU(inplace=True),
            )
          
    @staticmethod
    def _concat(x, out):
        # concatenate along channel axis
        return torch.cat((x, out), 1)        

    def forward(self, x):
        if 1==self.benchmodel:
            x1 = x[:, :(x.shape[1]//2), :, :]
            x2 = x[:, (x.shape[1]//2):, :, :]
            out = self._concat(x1, self.banch2(x2))
        elif 2==self.benchmodel:
            out = self._concat(self.banch1(x), self.banch2(x))

        return channel_shuffle(out, 2)


class ShuffleNetV2(nn.Module):
    def __init__(self,heads, head_conv, n_class=1000, input_size=224, width_mult=1.):
        super(ShuffleNetV2, self).__init__()
        self.inplanes = 24
        self.deconv_with_bias = False
        assert input_size % 32 == 0
        self.stage_repeats = [4, 8, 4]
        #self.stage_repeats = [2, 3, 2]
        # index 0 is invalid and should never be called.
        # only used for indexing convenience.
        if width_mult == 0.5:
            self.stage_out_channels = [-1, 24,  48,  96, 192, 1024]
        elif width_mult == 1.0:
            self.stage_out_channels = [-1, 24, 116, 232, 464, 1024]
        elif width_mult == 1.5:
            self.stage_out_channels = [-1, 24, 176, 352, 704, 1024]
        elif width_mult == 2.0:
            self.stage_out_channels = [-1, 24, 224, 488, 976, 2048]
        else:
            raise ValueError(
                """{} groups is not supported for
                       1x1 Grouped Convolutions""".format(num_groups))

        # building first layer
        input_channel = self.stage_out_channels[1]
        self.conv1 = conv_bn(3, input_channel, 2)    
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1) 
        self.features = []
        # building inverted residual blocks
        for idxstage in range(len(self.stage_repeats)):
            #self.features.append(SELayer(input_channel))
            numrepeat = self.stage_repeats[idxstage]
            output_channel = self.stage_out_channels[idxstage+2]
            for i in range(numrepeat):
                if i == 0:
	            #inp, oup, stride, benchmodel):
                    self.features.append(InvertedResidual(input_channel, output_channel, 2, 2))
                else:
                    self.features.append(InvertedResidual(input_channel, output_channel, 1, 1))
                input_channel = output_channel
                self.inplanes = output_channel
                
                
        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)

        # consider here to add the last sevearal layers
        # building last several layers
        self.conv_last  = conv_1x1_bn(input_channel, self.stage_out_channels[-1])
        self.inplanes =  self.stage_out_channels[-1]
        #self.cem1 = nn.Sequential(
        #            SELayer(116),
        #            nn.Conv2d(116, 256, 1, 2, 0, bias=False),
        #            nn.BatchNorm2d(256),
        #            nn.ReLU(inplace=True)
        #            )
        self.cem2 = nn.Sequential(
                    #SELayer(232),
                    conv_1x1_bn(232,256), #prepare to cat
                    nn.BatchNorm2d(256),
                    nn.ReLU(inplace=True)
                    )
        self.cem3 = self._make_deconv_layer(
            1,
            [256],
            [1],
        )        
        self.cemse = SELayer(256)
        # add heads
        self.heads = heads
        for head in self.heads:
            classes = self.heads[head]
            if head_conv > 0:
                fc = nn.Sequential(
                  nn.Conv2d(self.inplanes, head_conv,
                    kernel_size=3, padding=1, bias=True),
                  nn.ReLU(inplace=True),
                  nn.Conv2d(head_conv, classes, 
                    kernel_size=1, stride=1, 
                    padding=0, bias=True))
                if 'hm' in head:
                    fc[-1].bias.data.fill_(-2.19)
                else:
                    fill_fc_weights(fc)
            else:
                fc = nn.Conv2d(self.inplanes, classes, 
                  kernel_size=1, stride=1, 
                  padding=0, bias=True)
                if 'hm' in head:
                    fc.bias.data.fill_(-2.19)
                else:
                    fill_fc_weights(fc)
            self.__setattr__(head, fc)

        # building last several layers
        # self.conv_last      = conv_1x1_bn(input_channel, self.stage_out_channels[-1])
        # self.globalpool = nn.Sequential(nn.AvgPool2d(int(input_size/32)))              
    
	    #   # building classifier
        # self.classifier = nn.Sequential(nn.Linear(self.stage_out_channels[-1], n_class))

    def _get_deconv_cfg(self, deconv_kernel, index):
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0
        elif deconv_kernel == 1:
            padding = 0
            output_padding = 1

        return deconv_kernel, padding, output_padding

        # self.deconv_layers = self._make_deconv_layer(
        #     3,
        #     [256, 256, 256],
        #     [4, 4, 4],
        # )
    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        assert num_layers == len(num_filters), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'
        assert num_layers == len(num_kernels), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'

        layers = []
        for i in range(num_layers):
            kernel, padding, output_padding = \
                self._get_deconv_cfg(num_kernels[i], i)

            planes = num_filters[i]
            #layers.append(SELayer(self.inplanes))
            layers.append(
                nn.ConvTranspose2d(
                    in_channels=self.inplanes,
                    out_channels=planes,
                    kernel_size=kernel,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=self.deconv_with_bias))
            layers.append(nn.BatchNorm2d(planes))
            layers.append(nn.ReLU(inplace=True))
            self.inplanes = planes

        return nn.Sequential(*layers)
    def forward(self, x):
        #import pdb; pdb.set_trace()
        x = self.conv1(x)
        x = self.maxpool(x)
        #x = self.features(x)
        #import pdb; pdb.set_trace()
        for k, feat in enumerate(self.features):
          x = feat(x)
          if k == 3:
            cem1_input = x
          if k == 11:
            cem2_input = x
        #   if k == 15:
        #     cem3_input = x
        x = self.conv_last(x)
        cem3_input = x
        #import pdb; pdb.set_trace()
        #cem1_output = self.cem1(cem1_input)
        cem2_output = self.cem2(cem2_input)
        cem3_output = self.cem3(cem3_input)
        #x = cem1_output+cem2_output+cem3_output
        x = cem2_output+cem3_output
        x = self.cemse(x)
        ret = {}
        
        for head in self.heads:
            ret[head] = self.__getattr__(head)(x)
        return [ret]
        #x = self.conv_last(x)
        #x = self.globalpool(x)
        #x = x.view(-1, self.stage_out_channels[-1])
        #x = self.classifier(x)
        #return x
    
    def init_weights(self, pretrained=True):
        if pretrained:
            # print('=> init resnet deconv weights from normal distribution')
            for _, m in self.cem3.named_modules():
                if isinstance(m, nn.ConvTranspose2d):
                    nn.init.normal_(m.weight, std=0.001)
                    if self.deconv_with_bias:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
            #pretrained_state_dict = torch.load(pretrained)
            address = "/root/lbc/shufflenet-centernet/shufflenetv2_x1_69.402_88.374.pth.tar"
            pretrained_state_dict = torch.load(address)
            self.load_state_dict(pretrained_state_dict, strict=False)
            


def shufflenetv2(width_mult=1.):
    model = ShuffleNetV2(width_mult=width_mult)
    return model

# if __name__ == "__main__":
#     """Testing
#     """
#     inputs = torch.rand(1,3,32,32)
#     import pdb; pdb.set_trace()
#     model = ShuffleNetV2()
#     y = model(inputs)
#     print(model)

def get_shufflev2p_net(num_layers, heads, head_conv):
  model = ShuffleNetV2(heads, head_conv)
  model.init_weights( pretrained=True)
  
  return model