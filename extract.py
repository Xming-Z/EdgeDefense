import onnx


onnx.utils.extract_model('/data/Newdisk/zhaoxiaoming/steal/model_split_encryption/onnx/Military/Alexnet_Military.onnx',
                         '/data/Newdisk/zhaoxiaoming/steal/model_split_encryption/splited_model/alexnet3.onnx',
                         ['40'], ['output_data'])
#i-14 14-19 19-o

#onnx.utils.extract_model('/data/Newdisk/zhaoxiaoming/project_model_stealing/model_split_encryption/dp_onnx/imagenet_100/vgg16_imagenet_100.onnx',
#                         '/data/Newdisk/zhaoxiaoming/project_model_stealing/model_split_encryption/dp_split_model/imagenet_100/vgg16/vgg16_2.onnx',
#                        ['75'], ['output_data'])
#i-25 25-39 39-o

#onnx.utils.extract_model('/home/Newdisk/zhaoxiaoming/project_model_stealing/model_split_encryption/onnx/Military/VGG16_imagenet_100-299.onnx',
#                          '/home/Newdisk/zhaoxiaoming/project_model_stealing/model_split_encryption/splited_model/Military/vgg16/vgg16_1.onnx',
#                         ['input_data'], ['75'])
#i-39 39-54 54-o

#onnx.utils.extract_model('/home/Newdisk/zhaoxiaoming/project_model_stealing/model_split_encryption/onnx/Military/resnet50_Military.onnx',
#                          '/home/Newdisk/zhaoxiaoming/project_model_stealing/model_split_encryption/splited_model/Military/resnet50/resnet50_model3.onnx',
#                          ['439'], ['output_data'])
#i-355 355-439 439-o

#onnx.utils.extract_model('/home/Newdisk/zhaoxiaoming/project_model_stealing/model_split_encryption/onnx/Military/GoogleNet_Military.onnx',
#                          '/home/Newdisk/zhaoxiaoming/project_model_stealing/model_split_encryption/splited_model/Military/googlenet/googlenet2.onnx',
#                          ['559'], ['output_data'])

#i-417 417-518 518-o

#onnx.utils.extract_model('/data/Newdisk/zhaoxiaoming/object_detection/model/ssd_model/dp_onnx/ssd.onnx',
#                          '/data/Newdisk/zhaoxiaoming/object_detection/dp_split_model/ssd/ssd2.onnx',
#                          ['93'],['output','277'])
						  
#onnx.utils.extract_model('/data/Newdisk/zhaoxiaoming/object_detection/model/frcnn_model/dp_onnx/frcnn.onnx',
#                          '/data/Newdisk/zhaoxiaoming/object_detection/dp_split_model/faster_rcnn/faster_rcnn1.onnx',
#                         ['images'],['948','1093','474'])
						  
#onnx.utils.extract_model('/data/Newdisk/zhaoxiaoming/object_detection/model/frcnn_model/dp_onnx/frcnn.onnx',
#                          '/data/Newdisk/zhaoxiaoming/object_detection/dp_split_model/faster_rcnn/faster_rcnn2.onnx',
#                         ['1093','474'],['output','1122'])