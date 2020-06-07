import torch
from models.with_mobilenet import PoseEstimationWithMobileNet #my particular net architecture
from modules.load_state import load_state
from torch2trt import torch2trt #import library
import time

#HERE IT IS HOW COMPILE AND SAVE A MODEL
checkpoint_path='/home/nvidia/Documents/poseFINAL/checkpoints/body.pth' #your trained weights path

net = PoseEstimationWithMobileNet()#my particular net istance
checkpoint = torch.load(checkpoint_path, map_location='cuda')
load_state(net, checkpoint)#load your trained weights path
net.cuda().eval()

data = torch.rand((1, 3, 256, 344)).cuda()#initialize a random tensor with the shape of your input data

#model_trt = torch2trt(net, [data]) #IT CREATES THE COMPILED VERSION OF YOUR MODEL, IT TAKES A WHILE

#torch.save(model_trt.state_dict(), 'net_trt.pth') #TO SAVE THE WEIGHTS OF THE COMPILED MODEL WICH ARE DIFFERENT FROM THE PREVIOUS ONES


#HERE IT IS HOW TO UPLOAD THE MODEL ONCE YOU HAVE COMPILED IT LIKE IN MY CASE THAT I HAVE ALREADY COMPILED IT

from torch2trt import TRTModule #import a class

model_trt = TRTModule() #the compiled model istance

model_trt.load_state_dict(torch.load('net_trt.pth')) #load the compiled weights in the compiled model

#HERE IS HOW TO COMPARE THE TWO MODELS AND SEE THE 2 VELOCITIES, YOU SHOULD GET A TINY ERROR AND SEE THAT THE COMPILED NET IS MUCH FASTER
print(data)

now = time.time()
output_trt = model_trt(data)
print(time.time()-now)
now = time.time()
output = net(data)
print(time.time()-now)

print(output_trt[0])
print(output[0])
output_trt=output_trt[0]
output=output[0]
print(output.flatten()[0:10])
print(output_trt.flatten()[0:10])
print('max error: %f' % float(torch.max(torch.abs(output - output_trt))))
