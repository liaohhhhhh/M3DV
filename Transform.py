import numpy as np

class Transform:

	def __call__(self, arr, aux=None):

		shape = arr.shape
		angle = np.random.randint(4, size=3)
		arr_ret = rotation(arr, angle=angle)
		axis = np.random.randint(4) - 1
		arr_ret = reflection(arr_ret, axis=axis)
		#move = np.random.randint(4, size=3)
		#arr_ret = crop(arr_ret, move=move)
		if aux is not None:
			aux_ret = rotation(aux_ret, angle=angle)
			aux_ret = reflection(aux_ret, axis=axis)
			return arr_ret, aux_ret
		return arr_ret

def rotation(array, angle):
		X = np.rot90(array, angle[0], axes=(0, 1))  # rotate in X-axis
		Y = np.rot90(X, angle[1], axes=(0, 2))  # rotate in Y'-axis
		Z = np.rot90(Y, angle[2], axes=(1, 2))  # rotate in Z"-axis
		return Z

def reflection(array, axis):
	if axis != -1:
		ref = np.flip(array, axis)
	else:
		ref = np.copy(array)
	return ref

def crop(array, move):
	result = np.zeros((array.shape[0]+3,array.shape[1]+3,array.shape[2]+3))
	result[0:array.shape[0],0:array.shape[1],0:array.shape[2]] = array
	r = result[0+move[0]:array.shape[0]+move[0],0+move[1]:array.shape[1]+move[1],move[2]:array.shape[2]+move[2]]
	return r




