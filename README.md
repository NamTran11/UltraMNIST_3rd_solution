# UltraMNIST_3rd_solution
This repository is the 3rd solution for UltraMNIST Classification Challenge in Kaggle https://www.kaggle.com/competitions/ultra-mnist

### For data generation
```
- cd data_generator
- unzip digit-recognizer.zip
- python gen.py configs/your_config.yaml
// create your config informations in data_generator/configs/your_config.yaml
// after generation process, data will be save at data_generator/outputs
```

### For training new yolov5 model
```
- git clone https://github.com/ultralytics/yolov5
// update your dataset_path at yolov5_asset/ultra_mnist_fold0.yaml
- cp ./yolov5_asset/* ./yolov5
- cd yolov5
- python train.py --img 1024 --batch 8 --epochs 100 --data ./ultra_mnist_fold0.yaml --hyp ./hyp_first_train.yaml --cfg ./yolov5x.yaml --weights '' --name yolov5x_fold0
```

### For batch_size inference
```
// update your weights, dataset, save_result_dir in yolov5/detect.py
- cd yolov5
- python detect.py
```

### pretrained weights and submission result
comming soon

### References
Many thanks for https://github.com/ultralytics/yolov5
