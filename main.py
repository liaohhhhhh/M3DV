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

BATCHSIZE = 5
train_datas = MyDataset.MyTrainDataset()
train_loader1 = DataLoader(dataset=train_datas,batch_size=BATCHSIZE,shuffle=True)
train_loader2 = DataLoader(dataset=train_datas,batch_size=BATCHSIZE,shuffle=True)

train_transform_datas = MyDataset.MyTrainDataset(transform = Transform.Transform())
train_loader3 = DataLoader(dataset=train_transform_datas,batch_size=BATCHSIZE,shuffle=True)

val_datas = MyDataset.MyValDataset()
val_loader = DataLoader(dataset=val_datas,batch_size=1,shuffle=False)

test_datas = MyDataset.MyTestDataset()
test_loader = DataLoader(dataset=test_datas,batch_size=BATCHSIZE,shuffle=False)

device = torch.device('cuda')

model = DenseNet.DenseNet(growthRate=52, depth=10, reduction=0.25, nClasses=1, bottleneck=True).to(device)
#model = Net.resnet3d(1,1).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0)
#optimizer = torch.optim.RMSprop(model.parameters(),lr=1e-3,alpha=0.9)
#scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',factor=0.1,patience=7,verbose=True,min_lr=1e-5)
criterion = nn.BCELoss().cuda()
#criterion = Myloss.My_loss().cuda()

recode = []

for m in model.modules():
	if isinstance(m, nn.BatchNorm3d):
		nn.init.constant_(m.weight, 1)
		nn.init.constant_(m.bias, 0)
	if isinstance(m, (nn.Conv3d, nn.Linear)):
		nn.init.xavier_normal_(m.weight,gain=1)
		#nn.init.kaiming_normal_(m.weight, mode='fan_in')

minscore = 1
last = 0
bestcorrect = 0
time = 3
for epoch in range(30):
	sum_loss = 0
	alpha = 0.5
	model.train()
	#model.apply(freeze_bn)
	for (x1,y1),(x2,y2),(x3,y3) in zip(train_loader1,train_loader2,train_loader3):

		correct = 0
		lam = torch.from_numpy(np.random.beta(alpha,alpha,x1.shape[0]))
		#print(lam)
		#lam = torch.from_numpy(np.ones((x1.shape[0])))

		x_datas = torch.zeros((x1.shape[0]*time,x1.shape[1],x1.shape[2],x1.shape[3],x1.shape[4])).cuda()
		y_datas = torch.zeros((y1.shape[0]*time)).cuda()
		for i in range(x1.shape[0]):
			x_datas[i*time] = (x1[i] * lam[i] + x2[i] * (1. - lam[i])).cuda()
			x_datas[time*i + 1] = (x1[i]).cuda()
			x_datas[time*i + 2] = (x3[i]).cuda()
			#x_datas[time*i + 3] = (x2[i]).cuda()
			y_datas[i*time] = (lam[i] * y1[i] + (1. - lam[i]) * y2[i]).cuda()
			y_datas[time*i + 1] = (y1[i]).cuda()
			y_datas[time*i + 2] = (y3[i]).cuda()
			#y_datas[time*i + 3] = (y2[i]).cuda()

	# 	# x_datas = torch.zeros((x1.shape[0],x1.shape[1],x1.shape[2],x1.shape[3],x1.shape[4])).cuda()
	# 	# y_datas = torch.zeros((y1.shape[0])).cuda()
	# 	# for i in range(x1.shape[0]):
	# 	# 	x_datas[i] = (x1[i] * lam[i] + x2[i] * (1. - lam[i])).cuda()
	# 	# 	y_datas[i] = (lam[i] * y1[i] + (1. - lam[i]) * y2[i]).cuda()

		optimizer.zero_grad()
		outputs = model(x_datas).reshape((y_datas.shape))

		loss = criterion(outputs,y_datas)
		print('\nloss:',loss)
		
		loss.backward()
		optimizer.step()
		sum_loss += loss.item()

		del outputs
		del x_datas
		del y_datas
	average = sum_loss / math.ceil(train_datas.__len__()/BATCHSIZE)
	print('[epoch:%d,loss:%.03f]' %
		(epoch + 1, average))

	#scheduler.step(average)
	
	model.eval()
	print("VAL_TEST:")
	val_correct = 0
	for inputs,labels in val_loader:
		inputs = inputs.cuda()
		labels = labels.cuda()
		with torch.no_grad():
			outputs = model(inputs)
		
		print('(',outputs,'/',labels,')')
		outputs[outputs>=0.5] = 1
		outputs[outputs<0.5]  = 0
		if outputs == labels:
			val_correct += 1
	print("val_correct:",val_correct,'/',val_correct/val_loader.__len__())
	print("///////////////////////////////////////////////////////////")

	# for name, param in model.named_parameters():
	# 	print(name, param)

	if val_correct >= bestcorrect:
		bestcorrect = val_correct
		torch.save(model.state_dict(),'net_params.pkl')

	if average < minscore:
		minscore = average
		last = 0
	else:
		last += 1
		if last >= 15 and val_correct == bestcorrect:
			break

# 	recode.append([epoch+1,average])

# Recode.Recode(recode[0],recode[1])
model.load_state_dict(torch.load('net_params0.631.pkl'))
#torch.save(model.state_dict(),'net_params.pkl')
del train_loader1, train_loader2

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
	del outputs
print(result)
MyDataset.Write_Result(result) 