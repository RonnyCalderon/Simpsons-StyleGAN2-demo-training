# Simpsons StyleGAN2 🚀🚀🚀🚀🌑

Homework#6 

## Tech 🖥️
TensorFlow 2.0
Stylegan2: https://github.com/NVlabs/stylegan2
One or more high-end NVIDIA GPUs, NVIDIA drivers, CUDA 10.0 toolkit and cuDNN 7.5. To reproduce the results reported in the paper, you need an NVIDIA GPU with at least 16 GB of DRAM.
We used GPU.Land V100 Tesla GPU 0.99$/h 

## Steps 🛵

1 - Install last Conda version.
2 - Clone this Repository: https://github.com/zzh8829/yolov3-tf2
3 - Run: 
```bash
pip install scipy==1.3.3
pip install requests==2.22.0
pip install Pillow==6.2.1
```

## Training 🔥🔥🔥🔥🔥

1- Get the images from Kaggle and unzip.

```bash
pip install kaggle
export KAGGLE_USERNAME="{KAGGLE_USER}"
export KAGGLE_KEY="{KAGGLE_TOKEN}"
kaggle datasets download kostastokis/simpsons-faces
unzip simpsons-faces.zip 
```

2- Run Resize.py to converting the images in 64x64 resolutions
```bash
python resize.py
```

3- Create TFRecord with 64x64 images
```bash
python dataset_tool.py create_from_images /home/ubuntu/stylegan2/custom /home/ubuntu/stylegan2/64x64
```

4- Execute the algorithm
```bash
nohup python run_training.py --num-gpus=1 --data-dir=/home/ubuntu/stylegan2/datasets --config=config-f --dataset=custom --mirror-augment=true
```

5- Wait for the results! :D

## Continue the last training. 🎉 🎉 🎉 🎉 

If you can continue the last training. You can following the next steps:

1- In your stylegan2-master/results/ and find the most recent checkpoint, something like:

```bash
network-snapshot-000160.pkl
```

2- then we gotta edit a couple variables in training_loop.py
```bash
resume_pkl = 'path/stylegan2/results/00001-stylegan2-1gpu-config-f/network-snapshot-000160.pkl',
resume_kimg  = 160.0,
```
Why 160.0? - Because we need to convert the kimg value ("000160") to a float, and plug it into resume_kimg. Resume_kimg important since it needs to know where to resume the learning rate curve thing.

4- Execute the training command again
```bash
nohup python run_training.py --num-gpus=1 --data-dir=/home/ubuntu/stylegan2/datasets --config=config-f --dataset=custom --mirror-augment=true
```

## Images 🌮 🌮 🌮 🌮 

Real Image
![Download file on Roboflow](https://github.com/RonnyCalderon/Simpsons-StyleGAN2-demo-training/blob/main/images/reals.png)

Fake Init
![Download file on Roboflow](https://github.com/RonnyCalderon/Simpsons-StyleGAN2-demo-training/blob/main/images/fakes_init.png)

First Iteration
![Download file on Roboflow](https://github.com/RonnyCalderon/Simpsons-StyleGAN2-demo-training/blob/main/images/fakes000000.png)

Third Iteration
![Download file on Roboflow](https://github.com/RonnyCalderon/Simpsons-StyleGAN2-demo-training/blob/main/images/fakes000320.png)

Fifth Iteration
![Download file on Roboflow](https://github.com/RonnyCalderon/Simpsons-StyleGAN2-demo-training/blob/main/images/fakes000640.png)

Last Iteration
![Download file on Roboflow](https://github.com/RonnyCalderon/Simpsons-StyleGAN2-demo-training/blob/main/images/fakes001440.png)
