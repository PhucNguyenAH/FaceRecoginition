import json

import matplotlib.pyplot as plt
import mxnet as mx
from mxnet import gluon, nd
from mxnet.gluon.model_zoo import vision
import numpy as np

# We set the context to CPU, you can switch to GPU if you have one and installed a compatible version of MXNet 
ctx = mx.cpu()
# We can load three the three models
densenet121 = vision.densenet121(pretrained=True, ctx=ctx)
mobileNet = vision.mobilenet0_5(pretrained=True, ctx=ctx)
resnet18 = vision.resnet18_v1(pretrained=True, ctx=ctx)

print(mobileNet)

print(mobileNet.features[0].params)

mx.test_utils.download('https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/doc/tutorials/onnx/image_net_labels.json')
categories = np.array(json.load(open('image_net_labels.json', 'r')))
print(categories[4])

filename = mx.test_utils.download('https://github.com/dmlc/web-data/blob/master/mxnet/doc/tutorials/onnx/images/dog.jpg?raw=true', fname='dog.jpg')
filename = 'dog.jpg'

image = mx.image.imread(filename)
plt.imshow(image.asnumpy())


def transform(image):
    resized = mx.image.resize_short(image, 224) #minimum 224x224 images
    cropped, crop_info = mx.image.center_crop(resized, (224, 224))
    normalized = mx.image.color_normalize(cropped.astype(np.float32)/255,
                                      mean=mx.nd.array([0.485, 0.456, 0.406]),
                                      std=mx.nd.array([0.229, 0.224, 0.225])) 
    # the network expect batches of the form (N,3,224,224)
    transposed = normalized.transpose((2,0,1))  # Transposing from (224, 224, 3) to (3, 224, 224)
    batchified = transposed.expand_dims(axis=0) # change the shape from (3, 224, 224) to (1, 3, 224, 224)
    return batchified

predictions = resnet18(transform(image)).softmax()
print(predictions.shape)