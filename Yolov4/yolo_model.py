import torch
from torch import nn
import torch.nn.functional as F
from tool.torch_utils import *
from yolo_submodule import *


class Yolov4(nn.Module):
	def __init__(self, yolov4conv137weight=None, n_classes=80, inference=False):
		super(Yolov4,self).__init__()
		output_ch = (4 + 1 + n_classes) * 3
		# backbone
		self.down1 = DownSample1()
		self.down2 = DownSample2()
		self.down3 = DownSample3()
		self.down4 = DownSample4()
		self.down5 = DownSample5()
		# neck
		self.neek = Neck(inference)
		# yolov4conv137
		if yolov4conv137weight:
			_model = nn.Sequential(self.down1, self.down2, self.down3, self.down4, self.down5, self.neek)
			pretrained_dict = torch.load(yolov4conv137weight)

			model_dict = _model.state_dict()
			# 1. filter out unnecessary keys
			pretrained_dict = {k1: v for (k, v), k1 in zip(pretrained_dict.items(), model_dict)}
			# 2. overwrite entries in the existing state dict
			model_dict.update(pretrained_dict)
			# 3. load the new state dict
			_model.load_state_dict(model_dict)
		# head
		self.head = Yolov4Head(output_ch, n_classes, inference)
	def forward(self, input):
		d1 = self.down1(input)
		d2 = self.down2(d1)
		d3 = self.down3(d2)
		d4 = self.down4(d3)
		d5 = self.down5(d4)

		x20, x13, x6 = self.neek(d5, d4, d3)
		output = self.head(x20, x13, x6)
		return output