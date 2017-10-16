import math
import utils
import sys
import numpy as np

class naive_bayers_classifier(object):
	def __init__(self, training_img, training_label, mode=0):
		self.training_img = training_img
		self.training_label = training_label
		self.mode = mode
		self.num_per_class = np.asarray([28*28*np.sum(self.training_label == i) for i in range(10)])
		self.p_c = (self.num_per_class/np.sum(self.num_per_class))
		print (self.p_c)
		self.log_p_c = np.log(self.p_c)

	def train(self):
		if self.mode == 0:
			self.p_x_c = np.zeros((28, 28, 32, 10))
			#print (training_img[0] // 32)
			for i in range(60000):
				# 32 bins
				num_x_c = self.training_img[i] // 8
				for j in range(28):
					for k in range(28):
						self.p_x_c[j, k, num_x_c[j, k], self.training_label[i]] += 1
			for i in range(28):
				for j in range(28):
						for k in range(10):
							binnn = self.p_x_c[i, j, :, k]
							nonzero_min = np.min(binnn[np.nonzero(binnn)])
							#print (nonzero_min)
							self.p_x_c[i, j, (binnn==0), k] += nonzero_min
							sum_bin = np.sum(self.p_x_c[i, j, :, k])
							self.p_x_c[i, j, :, k] /= sum_bin
							#min_bins = np.min(self.p_x_c[i, j, k, m])
			#print (np.sum(self.p_x_c))
			#print (self.p_x_c)
			self.log_p_x_c = np.log(self.p_x_c)
			#print (self.log_p_x_c)
		if self.mode == 1:
			self.p_x_c = np.zeros((28, 28, 2, 10))
			element_list = []
			for k in range(10):
				element = np.asarray([self.training_img[n] for n in range(60000) if self.training_label[n]==k])
				element_list.append(element)

			for row in range(28):
				for col in range(28):
					for k in range(10):
						element = element_list[k][:, row, col]
						#print (len(element_list[k]))
						self.p_x_c[row, col, 0, k] = np.sum(element) / len(element_list[k])
						#print (len(element_list[k]))
						#print (((element - self.p_x_c[row, col, 0, k])**2).shape)
						self.p_x_c[row, col, 1, k] = np.sum((element - self.p_x_c[row, col, 0, k])**2) / len(element_list[k])
						if self.p_x_c[row, col, 1, k] == 0:
							self.p_x_c[row, col, 1, k] = 100
						#print (row, col, k, self.p_x_c[row, col, 0, k])

	def log_gaussian_likelihood(self, mean, variace, x):
		return -0.5*np.log(2 * math.pi * variace) + -((x-mean) ** 2)/(2*variace)

	def test(self, testing_img, testing_label):
		if self.mode == 0:
			# images
			prediction_results = []
			for i in range(len(testing_img)):
				log_likelihood_image = np.asarray([0.0] * 10)
				# rows
				for row in range(28):
					# cols
					log_likelihood_row = np.asarray([0.0] * 10)
					for col in range(28):
						val = testing_img[i, row, col] // 8
						log_likelihood_row += self.log_p_x_c[row, col, val, :]
					log_likelihood_image += log_likelihood_row
					#print ("row %.2d: " % (row), testing_img[i, row, :], "log_likelihood:", log_likelihood_row)
					print ("row %.2d: " % (row), "log_likelihood:", log_likelihood_row+self.log_p_c)
				prediction = np.argmax(log_likelihood_image+self.log_p_c)
				prediction_results.append(prediction)	
				print ("Image [%d/10000], prediction: %d, label: %d" % (i+1, prediction, testing_label[i]))
			acc = np.sum(testing_label==prediction_results) / len(testing_label)
			print ("Fianl Accuracy: %.4f" % acc)

		if self.mode == 1:
			# images
			prediction_results = []
			for i in range(len(testing_img)):
				log_likelihood_image = np.asarray([0.0] * 10)
				# rows
				for row in range(28):
					# cols
					log_likelihood_row = np.asarray([0.0] * 10)
					for col in range(28):
						val = testing_img[i, row, col]
						log_likelihood_row += self.log_gaussian_likelihood(self.p_x_c[row, col, 0, :], self.p_x_c[row, col, 1, :], val)
					log_likelihood_image += log_likelihood_row
					#print ("row %.2d: " % (row), testing_img[i, row, :], "log_likelihood:", log_likelihood_row)
					print ("row %.2d: " % (row), "log_likelihood:", log_likelihood_row+self.log_p_c)
				prediction = np.argmax(log_likelihood_image+self.log_p_c)
				prediction_results.append(prediction)	
				print ("Image [%d/10000], prediction: %d, label: %d" % (i+1, prediction, testing_label[i]))
			acc = np.sum(testing_label==prediction_results) / len(testing_label)
			print ("Fianl Accuracy: %.4f, Error rate: %.4f" % (acc, 1-acc))



def main():
	data_path_dict = {}
	arguments = sys.argv
	#print (arguments)
	data_path_dict["training_img"] = arguments[1]
	data_path_dict["training_label"] = arguments[2]
	data_path_dict["testing_img"] = arguments[3]
	data_path_dict["testing_label"] = arguments[4]
	mode = int(arguments[5])

	loader = utils.mnist_loader(data_path_dict)
	'''
	print (loader.training_img.shape)
	print (loader.training_label.shape)
	print (loader.testing_img.shape)
	print (loader.testing_label.shape)
	'''
	classifier = naive_bayers_classifier(loader.training_img, loader.training_label, mode)
	classifier.train()
	classifier.test(loader.testing_img, loader.testing_label)

if __name__ == "__main__":
	main()