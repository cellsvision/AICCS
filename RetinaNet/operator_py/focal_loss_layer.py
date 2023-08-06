import mxnet as mx
import numpy as np
from ast import literal_eval


class FocalLoss(mx.operator.CustomOp):
    '''
    '''
    def __init__(self, alpha, gamma, normalize, is_pn):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.normalize = normalize

        self.eps = 1e-14
        self.is_pn = is_pn

    def forward(self, is_train, req, in_data, out_data, aux):
        '''
        Just pass the data.
        '''
        #print("#### indata 0")
        #print(in_data[0])
        #print("#### indata 1")
        #print(in_data[1])
        #print("#### indata 2")
        #print(in_data[2])
        self.assign(out_data[0], req[0], in_data[1])

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        '''
        Reweight loss according to focal loss.
        '''
        cls_target = mx.nd.reshape(in_data[2], (0, 1, -1))    # [batchsize, 1, num_anchors]
        if self.is_pn:
            cls_target_array = cls_target.asnumpy()
            cls_target_array[cls_target_array > 0] = 1
            cls_target = mx.nd.array(cls_target_array)
        p = mx.nd.pick(in_data[1], cls_target, axis=1, keepdims=True)    # [batchsize, 1, num_anchors]

        n_class = in_data[0].shape[1]

        u = 1 - p - (self.gamma * p * mx.nd.log(mx.nd.maximum(p, self.eps)))
        v = 1 - p if self.gamma == 2.0 else mx.nd.power(1 - p, self.gamma - 1.0)
        a = (cls_target > 0) * self.alpha + (cls_target == 0) * (1 - self.alpha)
        gf = v * u * a

        label_mask = mx.nd.one_hot(mx.nd.reshape(cls_target, (0, -1)), n_class,
                on_value=1, off_value=0)
        label_mask = mx.nd.transpose(label_mask, (0, 2, 1))
        #print('#### in_data')
        #print(in_data[1].shape)
        #print("#### label mask")
        #print(label_mask.shape)
        #print("#### gf")
        #print(gf.shape)

        g = (in_data[1] - label_mask) * gf
        g *= (cls_target >= 0)

        if self.normalize:
            g /= max(1.0, mx.nd.sum(cls_target > 0).asscalar())


        self.assign(in_grad[0], req[0], g)
        self.assign(in_grad[1], req[1], 0)
        self.assign(in_grad[2], req[2], 0)


@mx.operator.register("focal_loss")
class FocalLossProp(mx.operator.CustomOpProp):
    '''
    '''
    def __init__(self, alpha=0.25, gamma=2.0, normalize=True, is_pn=False):
        #
        super(FocalLossProp, self).__init__(need_top_grad=False)
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.normalize = bool(literal_eval(str(normalize)))
        self.is_pn = is_pn

    def list_arguments(self):
        return ['cls_pred', 'cls_prob', 'cls_target']

    def list_outputs(self):
        return ['cls_prob']

    def infer_shape(self, in_shape):
        out_shape = [in_shape[0], ]
        return in_shape, out_shape, []

    def create_operator(self, ctx, shapes, dtypes):
        return FocalLoss(self.alpha, self.gamma, self.normalize, self.is_pn)
