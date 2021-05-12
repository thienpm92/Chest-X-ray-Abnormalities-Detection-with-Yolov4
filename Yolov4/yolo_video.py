import sys
import cv2
import time
import torch
from torch import nn
import torch.nn.functional as F
from tool.torch_utils import *
from yolo_model import Yolov4
from tool.utils import load_class_names, plot_boxes_cv2
from tool.torch_utils import do_detect
import numpy as np




if __name__ == "__main__":
	if len(sys.argv) == 6:
		n_classes = int(sys.argv[1])
		weightfile = sys.argv[2]
		imgfile = sys.argv[3]
		height = int(sys.argv[4])
		width = int(sys.argv[5])
	else:
		print('Usage: ')
		print('python yolo_video.py num_classes weightfile imgfile H W')
	namesfile = None
	device =  torch.device('cuda' if torch.cuda.is_available() else 'npu')
	model = Yolov4(yolov4conv137weight=None, n_classes=n_classes, inference=True)
	pretrained_dict = torch.load(weightfile, map_location=device)
	model.load_state_dict(pretrained_dict)

	if device == torch.device('cuda'):
		use_cuda = True
		model.cuda()
	else:
		use_cuda = False

	if namesfile == None:
		if n_classes == 20:
			namesfile = 'data/voc.names'
		elif n_classes == 80:
			namesfile = 'data/coco.names'
		else:
			print("please give namefile")

	path = 'regular_street.avi'
	vid = cv2.VideoCapture(path)
	Nframe = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
	model.eval()
	t1 = time.time()
	for frame_id in range(0,Nframe):
		check,img = vid.read()
		if not check:
			break
		img = cv2.resize(img, (width, height))
		img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


		for i in range(1):  # This 'for' loop is for speed check
                        # Because the first iteration is usually longer
			boxes = do_detect(model, img, 0.5, 0.6, use_cuda)

		class_names = load_class_names(namesfile)
		img = plot_boxes_cv2(img, boxes[0], class_names = class_names)
		img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
		#cv2.imshow('result',img)
		#cv2.waitKey(1)
	t2 = time.time()
	print('time:',t2-t1)
	print('FPS:',Nframe/(t2-t1))