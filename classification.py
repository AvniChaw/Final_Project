#1/usr/bin/python3

import jetson.inference
import jetson.utils

import argparse

parser = argparse.Argumentparser()
parser.add_argument("filename", type=str, help="filename of the image to process")
parser.add_argument ("--network", type = str, default="detectnet", help = "modle to use can be: detectnet, resnet-18,ect. ( see --help for others) " )
opt = parser.pars_args ()
img = jetson.utils.loadImage(opt.filename)
net = jetson.inference.detectNet(opt.network)
class_idx, confidence = net.Classify(img)
class_desc = net.GetCLassDesc(class_idx) 
for item in class_desc: 
  if(item == "Vegetable!"): 
    print("image is recognized as '{:s}' (class #{:d}) with {:f}% confidence".format(class_desc, class_idx, confidence * 100))
  if(item == "Fruit!"):
     print("image is recognized as '{:s}' (class #{:d}) with {:f}% confidence".format(class_desc, class_idx, confidence * 100))
                                          
