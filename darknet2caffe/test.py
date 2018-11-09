import sys  
sys.path.append('/workspace/caffe_szx/python')  
import caffe  
import numpy as np  
import cv2
import os

result_dir = '/workspace/caffe_szx/result'

# caffe.set_mode_cpu()
caffe.set_device(0)
caffe.set_mode_gpu()

caffemodel = os.path.join(result_dir, 'V3.caffemodel')
deploy = os.path.join(result_dir, 'V3.prototxt')
img_filepath = "/workspace/caffe_szx/test/cardet1_1.jpg"


net = caffe.Net(deploy,  
                caffemodel,  
                caffe.TEST)


# transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
# transformer.set_transpose('data', (2,0,1))
# transformer.set_raw_scale('data', 255)
# transformer.set_channel_swap('data', (2,1,0))

img = cv2.imread(img_filepath)
img_resize = cv2.resize(img, (640, 640))
img_normal = img_resize / 255.0
img_rgb = img_normal[:, :, ::-1]
img_input = np.transpose(img_rgb, (2, 0, 1))
net.blobs['data'].data[...] = np.expand_dims(img_input, axis=0)
# print(net.blobs['data'].data.shape)

outputs = net.forward()

# print(outputs.keys())
# np.save('./yolo_output.npy', outputs)

for layername in net.outputs:
	print(net.blobs[layername].data.shape)
	np.save(os.path.join(result_dir, '%s.npy' % layername), net.blobs[layername].data)


# for k,v in net.params.items():
# 	print(k,v[0].data.shape)

# for k,v in net.blobs.items():
# 	print(k,v.data.shape)

# print(dir(net))
# print(net.outputs)  # ['layer106-conv', 'layer82-conv', 'layer94-conv']
# print(type(net.blobs['layer82-conv'].data))


# img = cv2.imread(img_filepath)
# img_resize = cv2.resize(img, (640, 640))
# img_input = img_resize / 255.0
# img_input = img_input[:, :, ::-1]
# net.blobs['data'].data[...] = img_input



