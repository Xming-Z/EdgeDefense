import onnx


onnx.utils.extract_model('Alexnet_Military.onnx',
                         'splited_model/alexnet3.onnx',
                         ['40'], ['output_data'])
#i-14 14-19 19-o

#onnx.utils.extract_model('vgg16_imagenet_100.onnx',
#                         '/vgg16/vgg16_2.onnx',
#                        ['75'], ['output_data'])
#i-25 25-39 39-o

#onnx.utils.extract_model('VGG16_imagenet_100-299.onnx',
#                          'vgg16/vgg16_1.onnx',
#                         ['input_data'], ['75'])
#i-39 39-54 54-o

#onnx.utils.extract_model('resnet50_Military.onnx',
#                          '/resnet50/resnet50_model3.onnx',
#                          ['439'], ['output_data'])
#i-355 355-439 439-o

#onnx.utils.extract_model('GoogleNet_Military.onnx',
#                          '/googlenet/googlenet2.onnx',
#                          ['559'], ['output_data'])

#i-417 417-518 518-o

#onnx.utils.extract_model('ssd.onnx',
#                          '/split_model/ssd/ssd2.onnx',
#                          ['93'],['output','277'])
						  
#onnx.utils.extract_model('frcnn.onnx',
#                          '/split_model/faster_rcnn/faster_rcnn1.onnx',
#                         ['images'],['948','1093','474'])
						  
