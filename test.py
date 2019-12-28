import MyDataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch
import DenseNet
import torch.nn as nn
import math
from tqdm import tqdm 
import numpy as np
import Transform

def freeze_bn(m):
	if isinstance(m,nn.BatchNorm3d):
		m.eval()

BATCHSIZE = 1

test_datas = MyDataset.MyTestDataset()
test_loader = DataLoader(dataset=test_datas,batch_size=BATCHSIZE,shuffle=False)

device = torch.device('cuda')

model = DenseNet.DenseNet(growthRate=52, depth=10, reduction=0.25, nClasses=1, bottleneck=True).to(device)

minscore = 1
last = 0
bestcorrect = 0
time = 3

model_dir = input('--model_params_dir:')
outputs_dir = input('--outputs_dir:')

model.load_state_dict(torch.load(model_dir))

model.eval()
result = []
i = 0
for data_test in test_loader:
	print(i)
	i+=1
	data_test = data_test.cuda()
	with torch.no_grad():
		outputs = model(data_test)
	for score in outputs:
		result.append(score)
print(result)
MyDataset.Write_Result(result,outputs_dir) 