from torch.utils.data import Dataset,DataLoader
import Transform
import numpy as np
from random import sample
SIZE = 32
class MyTrainDataset(Dataset):
	def __init__(self, txt_path = 'train_val.csv', transform = None, target_transform = None):
		fh = open(txt_path,'r')
		datas = []
		line = fh.readline()
		for line in fh:
			line = line.rstrip()
			words = line.split(',')
			datas.append(('./train_val/'+words[0]+'.npz',np.float32(words[1])))

		# self.val_datas = sample(datas,len(datas)//5)
		# for name in self.val_datas:
		# 	datas.remove(name)

		self.datas = datas
		self.transform = transform
		self.target_transform = target_transform
		self.size = SIZE

	def __getitem__(self, index):
		fn, label = self.datas[index]
		data = np.load(fn)
		k = int(self.size/2)
		voxel = data['voxel'][50-k:50+k,50-k:50+k,50-k:50+k]
		seg = data['seg'][50-k:50+k,50-k:50+k,50-k:50+k].astype(np.float32)
		if self.transform is not None:
			voxel = self.transform(voxel)
		
		# mean = np.mean(voxel)
		# std = np.std(voxel)
		# voxel = (voxel - mean) / std

		_min = 0
		_max = 255
		voxel = (voxel - _min) / (_max - _min)
		#voxel = voxel * seg
		voxel = voxel.astype(np.float32).reshape([1,SIZE,SIZE,SIZE])
		return voxel,label

	def __len__(self):
		return len(self.datas)

	def val_len(self):
		return len(self.val_datas)

	def get_val(self, index):
		fn, label = self.val_datas[index]
		data = np.load(fn)
		k = int(self.size/2)
		voxel = data['voxel'][50-k:50+k,50-k:50+k,50-k:50+k]
		seg = data['seg'][50-k:50+k,50-k:50+k,50-k:50+k].astype(np.float32)
		
		# mean = np.mean(voxel)
		# std = np.std(voxel)
		# voxel = (voxel - mean) / std

		_min = 0
		_max = 255
		voxel = (voxel - _min) / (_max - _min)
		#voxel = voxel * seg
		voxel = voxel.astype(np.float32).reshape([-1,1,SIZE,SIZE,SIZE])

		# if index == len(self.val_datas) - 1:
		# 	self.val_datas = sample(self.datas,len(self.datas)//5)
		return voxel,label


class MyValDataset(Dataset):
	def __init__(self, txt_path = 'Val.csv', transform = None, target_transform = None):
		fh = open(txt_path,'r')
		datas = []
		line = fh.readline()
		for line in fh:
			line = line.rstrip()
			words = line.split(',')
			datas.append(('./train_val/'+words[0]+'.npz',np.float32(words[1])))

		self.datas = datas
		self.transform = transform
		self.target_transform = target_transform
		self.size = SIZE

	def __getitem__(self, index):
		fn, label = self.datas[index]
		data = np.load(fn)
		k = int(self.size/2)
		voxel = data['voxel'][50-k:50+k,50-k:50+k,50-k:50+k]
		seg = data['seg'][50-k:50+k,50-k:50+k,50-k:50+k].astype(np.float32)
		if self.transform is not None:
			voxel = self.transform(voxel)
		
		# mean = np.mean(voxel)
		# std = np.std(voxel)
		# voxel = (voxel - mean) / std

		_min = 0
		_max = 255
		voxel = (voxel - _min) / (_max - _min)
		#voxel = voxel * seg
		voxel = voxel.astype(np.float32).reshape([1,SIZE,SIZE,SIZE])
		# print(voxel)
		# print('---------------------')
		# print(label)
		return voxel,label

	def __len__(self):
		return len(self.datas)

class MyTestDataset(Dataset):
	def __init__(self, txt_path= 'test.csv', transform = None, target_transform = None):
		
		fh = open(txt_path,'r')
		datas = []
		line = fh.readline()
		for line in fh:
			line = line.rstrip()
			words = line.split(',')
			datas.append(('./test/'+words[0]+'.npz'))

		self.datas = datas
		self.transform = transform
		self.target_transform = target_transform
		self.size = SIZE

	def __getitem__(self, index):
		fn = self.datas[index]
		data = np.load(fn)
		#print(fn)
		k = int(self.size/2)
		voxel = data['voxel'][50-k:50+k,50-k:50+k,50-k:50+k]
		seg = data['seg'][50-k:50+k,50-k:50+k,50-k:50+k].astype(np.float32)
		
		# mean = np.mean(voxel)
		# std = np.std(voxel)
		# voxel = (voxel - mean) / std

		_min = 0
		_max = 255
		voxel = (voxel - _min) / (_max - _min)
		#voxel = voxel * seg
		voxel = voxel.astype(np.float32).reshape([1,SIZE,SIZE,SIZE])
		return voxel

	def __len__(self):
		return len(self.datas)

def Write_Result(result,path):

	file = open('test.csv','r')
	datas = []
	for line in file:
		line = line.rstrip()
		words = line.split(',')
		datas.append([words[0],words[1]])

	for i in range(1,len(datas)):
		datas[i][1] = float(result[i-1])
	file.close()

	if path != '':
		file = open(path + '/submission.csv','w')
	else:
		file = open('submission.csv','w')

	datas[0][0] = 'Id'
	datas[0][1] = 'Predicted'

	for i in range(len(datas)):
		file.write(datas[i][0]+','+str(datas[i][1])+'\n')
	file.close()

if __name__ == '__main__':
	datas = MyValDataset()
	testloader = DataLoader(dataset=datas,batch_size=1,shuffle=False)

	for epoch in range(5):
		for data,label in testloader:
			print(label)
'''
			seg = inputs['seg'][0]
			temp = seg[50][50][50]
			XL = XH = 50
			YL = YH = 50
			ZL = ZH = 50
			while(temp == True):
				temp = seg[XL][50][50]
				XL -= 1
			temp = seg[50][50][50]
			while(temp == True):
				temp = seg[XH][50][50]
				XH += 1
			temp = seg[50][50][50]
			while(temp == True):
				temp = seg[YL][50][50]
				YL -= 1
			temp = seg[50][50][50]
			while(temp == True):
				temp = seg[YH][50][50]
				YH += 1
			temp = seg[50][50][50]
			while(temp == True):
				temp = seg[ZL][50][50]
				ZL -= 1
			temp = seg[50][50][50]
			while(temp == True):
				temp = seg[ZH][50][50]
				ZH += 1
			print(i,"X,Y,Z:(",XL,",",XH,",",YL,",",YH,",",ZL,",",ZH,")")
'''
