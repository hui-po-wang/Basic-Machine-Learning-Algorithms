import os
import struct
import numpy as np

class mnist_loader(object):
	def __init__(self, data_path_dict):
		fname_training_img = data_path_dict["training_img"]
		print (fname_training_img)
		fname_training_label = data_path_dict["training_label"]
		fname_testing_img = data_path_dict["testing_img"]
		fname_testing_label = data_path_dict["testing_label"]

		with open(fname_training_img, 'rb') as ftnimg:
			magic, num, rows, cols = struct.unpack(">IIII", ftnimg.read(16))
			self.training_img = np.fromfile(ftnimg, dtype=np.uint8).reshape(num, rows, cols)

		with open(fname_testing_img, 'rb') as fttimg:
			magic, num, rows, cols = struct.unpack(">IIII", fttimg.read(16))
			self.testing_img = np.fromfile(fttimg, dtype=np.uint8).reshape(num, rows, cols)

		with open(fname_training_label, 'rb') as ftnlabel:
			magic, num = struct.unpack(">II", ftnlabel.read(8))
			self.training_label = np.fromfile(ftnlabel, dtype=np.uint8)

		with open(fname_testing_label, 'rb') as fttlabel:
			magic, num  = struct.unpack(">II", fttlabel.read(8))
			self.testing_label = np.fromfile(fttlabel, dtype=np.uint8)