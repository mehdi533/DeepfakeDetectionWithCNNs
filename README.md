# Exploring Diffusion-Generated Image Detection Methods

Research done by: [Mehdi Abdallahi](https://www.linkedin.com/in/mehdi-abdallahi/?originalSubdomain=es)
<br>Supervisor: [Yuhang Lu](https://scholar.google.com/citations?hl=en&user=CtNglVsAAAAJ&view_op=list_works&sortby=pubdate)
<br>Professor: [Dr. Touradj Ebrahimi](https://scholar.google.com/citations?user=jt-UsrcAAAAJ&hl=en)

---

The content of this repository contains the material used for the research done during the second semester of my Master's at EPFL on the detection of diffusion-generated images, the code is adapted from the paper [CNN-generated images are surprisingly easy to spot... for now](https://arxiv.org/pdf/1912.11035.pdf) published by Wang et al. in 2020, and the code they used can be found in this [repository](https://github.com/peterwang512/CNNDetection). Click here for the [report](https://github.com/mehdi533/Deepfake-Detection/blob/main/Exploring_Diffusion_Generated_Image_Detection_Methods.pdf) and [presentation slides](https://github.com/mehdi533/Deepfake-Detection/blob/main/FinalPresentation.pdf) of this project. The trainings I did were done on the [izar](https://www.epfl.ch/research/facilities/scitas/hardware/izar/) cluster at EPFL using slurm commands.

---

For this research, the main focus was to gather useful insights for the generalization of entire face synthesis detection. To address this task, a state-of-the-art synthesized image detection method was first established as a baseline. In order to train and assess the performances of a wide range of models, a convenient pipeline for training, validation and evaluation was implemented. This allowed to move onto the next step: the exploration of large pre-trained vision models. More than 15 CNN architectures were explored, thanks to which interesting insights on the generalization of entire face synthesis classification were gathered.

---

### Getting started

```python
git clone https://github.com/mehdi533/Deepfake-Detection
cd Deepfake-Detection/
```
The required libraries are in the requirements.txt file and can be installed using the command
```python
pip install -r requirements.txt
```
---

### Datasets

The datasets in the same format used for this study are available on the [MMSPG page](https://www.epfl.ch/labs/mmspg/downloads/ai-synthesized-human-face-dataset/).

The images of a specific generator (e.g. ProGAN) are all located in the same folder, the correct images for training, validation, and testing, are selected by loading the lists of images in the metadata folder. The emplacement of the Metadata folder has to be passed by argument, by default it is set as Deepfake-Detection/dataset/metadata.

<br>The format in which the metadata is stored has to be similar to this:

```
├── Metadata
│   ├── train
│   │   ├── 0_real
│   │   │   ├── X_real_train_list.txt
│   │   │   └── Y_real_train_list.txt
│   │   ├── 1_fake
│   │   │   ├── A_fake_train_list.txt
│   │   │   ├── B_fake_train_list.txt
│   │   │   ├-- ...
│   │   │   
│   ├── val
│   │   ├── 0_real
│   │   │   ├── X_real_val_list.txt
│   │   │   └── Y_real_val_list.txt
│   │   ├── 1_fake
│   │   │   ├── A_fake_val_list.txt
│   │   │   ├── B_fake_val_list.txt
│   │   │   ├-- ...
│   │   │   
│   ├── test
│   │   ├── 0_real
│   │   │   ├── X_real_test_list.txt
│   │   │   └── Y_real_test_list.txt
│   │   ├── 1_fake
│   │   │   ├── A_fake_test_list.txt
│   │   │   ├── B_fake_test_list.txt
│   │   │   ├-- ...
│   │   │      

```

X,Y,A, and B represent names of folders in which images are stored, they need to correspond for the fetching of the image path to be done properly.

<br>For the images, they have to be stored like this:
```
├── data
│   ├── ProGAN
│   │   ├── 00000.png
│   │   ├── ...
│   ├── DDIM
│   │   ├── 00000.png
│   │   ├── ...
│   ├── CelebA-HQ
│   │   ├── 00000.png
│   │   ├── ...
```

> It is easy to add new datasets for training/testing. When testing on datasets that do not have the same name as the ones used for this study, please adapt the "list_data" in the util.py file. Your models will then be tested on the datasets present in this list.

---

### Train a model

![image](https://github.com/mehdi533/VisiumHackaton/assets/113531778/44be5b6e-2ca5-463d-9b36-cb394223f3cc)

To train a model, there are a lot of options available. 

The possible flag options we propose are:
- `--checkpoints_dir`: models trained will be saved in this directory
- `--name`: the name of the experiment, the results will be saved in: checkpoints_dir, filename will be <arch>_<name>_<models>
- `--arch`: architecture of the model (check list below for available ones)
- `--intermediate`: adds a fully connected layer in the classifier (use when training with frozen backbone)
- `--intermediate_dim`: the dimension of the added layer
- `--freeze`: option to freeze the backbone of the model
- `--pre_trained`: use the model with pre trained weights
- `--models`: models/generators on which the model will be trained (e.g. CelebA-HQ,ProGAN,DDIM)
- `--multiply_real`: to upsample the real class, the amount of real images will be multiplied by this amount
- `--batch_size`: the batch size
- `--dataroot`: the path to the folder containing the different images from the different generators (e.g. the path to "data" presented when showing how the images should be stored) 
- `--metadata`: the path to the folder containing the metadata as shown above.
- `--num_threads`: the number of threads to use 
- `--cropping`: crop images in random patches
- `--compr_prob`: the percentage of images to be pre processed with compression
- `--blur_prob`: the percentage of images to be pre processed with blurring

You simply have to run a command with your chosen options, use the different flags to have the different modes of training (fine-tuning by default, freeze and add a fully connected layer for frozen backbone, set pre-trained to false to train newly initialized layers):

```python
# ResNet50 (Fine-tuning)
python train.py --arch res50 --name 1106 --pre_trained --multiply_real 2 --batch_size 256 --blur_prob 0.3 --models real,DDIM,ProGAN --checkpoints_dir ./checkpoints/ --dataroot ./dataset/ --metadata ./dataset/metadata/

# Swin Tiny (Frozen backbone)
python train.py --arch swin_tiny --name 1206 --freeze --intermediate --pre_trained --batch_size 256 --blur_prob 0.1 --models CelebAHQ,FFpp0,FFpp1,ProGAN --checkpoints_dir ./checkpoints/ --dataroot ./dataset/ --metadata ./dataset/metadata/

# Big Transfer (Training newly initialized layers)
python train.py --arch bit --name 1206 --no-pre_trained --batch_size 256 --blur_prob 0.1 --models CelebAHQ,FFpp0,FFpp1,ProGAN --checkpoints_dir ./checkpoints/ --dataroot ./dataset/ --metadata ./dataset/metadata/
```

The details of the implementation (optimizer, learning rate scheduler...) can be found in the [report](XXX). 

---

### Testing

To test a model, the way of doing is similar, the options listed below are available:
- `--path`: path to the model (.pth) or to multiple models if using the model ensemble
- `--name`: the name of the experiment, the results will be saved in 
- `--intermediate`: adds a fully connected layer in the classifier (use when training with frozen backbone)
- `--intermediate_dim`: the dimension of the added layer
- `--batch_size`: the batch size
- `--dataroot`: the path to the folder containing the different images from the different generators (e.g. the path to "data" presented when showing how the images should be stored) 
- `--metadata`: the path to the folder containing the metadata as shown above.
- `--num_threads`: the number of threads to use
- `--meta_model`: option to train a model on the validation set to optimize the weights for the vote of the models (LR or kNN).
- `--models`: models/generators on which the meta model will be trained (e.g. CelebA-HQ,ProGAN,DDIM)

To change the directory to save the results, check the util.py file. The intermediate and intermediate_dim are mendatory to use when testing a model that was trained using those flags. The default meta model is none, to choose one put LR for linear regression or kNN for k Nearest Neighbors in the --meta_model flag, if you want to change the numbers of neighbors, the distance metric or other, you will have to do so manually in the model_ensemble.py file.

To evaluate a model, you simply have to run a command with your chosen options:
```python
# Simple evaluation of a model
python eval.py --name testFFpp3 --batch_size 256 --path swin_tiny_0506_FFpp3/model_epoch_best.pth

# With Model Ensemble
python eval.py --name LRtest --batch_size 256 --meta_model LR --path trained/swin_tiny_Forensics --models FFpp0,FFpp1,FFpp2,FFpp3 --num_threads 8
```
---

### Models available 

The different architectures that you can train on using this code are the following, the value to pass with the `--arch` flag is given first:
- `res50` [ResNet50](https://pytorch.org/vision/main/models/generated/torchvision.models.resnet50.html)
- `vgg16` [VGG16](https://pytorch.org/vision/main/models/generated/torchvision.models.vgg16.html)
- `efficient_b0` [EfficientNet b0](https://pytorch.org/vision/main/models/generated/torchvision.models.efficientnet_b0.html)
- `efficient_b4` [EfficientNet b4](https://pytorch.org/vision/main/models/generated/torchvision.models.efficientnet_b4.html)
- `bit` [Big Transfer](https://huggingface.co/google/bit-50)
- `vit_base` [Vision Transformer (base size)](https://huggingface.co/google/vit-base-patch16-224)
- `vit_large` [Vision Transformer (large size)](https://huggingface.co/google/vit-large-patch16-224)
- `deit_small`[Distilled Data-efficient Image Transformer (small-sized model)](https://huggingface.co/facebook/deit-small-distilled-patch16-224)
- `deit_base` [Distilled Data-efficient Image Transformer (base-sized model)](https://huggingface.co/facebook/deit-base-distilled-patch16-224)
- `coatnet` [CoAtNet](https://huggingface.co/timm/coatnet_0_rw_224.sw_in1k)
- `resnext` [ResNeXt](https://huggingface.co/timm/resnext101_32x8d.tv_in1k)
- `beit` [BEiT (base-sized model)](https://huggingface.co/microsoft/beit-base-patch16-224)
- `convnext` [ConvNeXt](https://pytorch.org/vision/main/models/convnext.html)
- `regnet` [RegNetY-400MF](https://pytorch.org/vision/main/models/generated/torchvision.models.regnet_y_400mf.html#torchvision.models.regnet_y_400mf)
- `swin_tiny` [Swin Transformer (tiny version)](https://huggingface.co/microsoft/swin-tiny-patch4-window7-224)
- `swin_base` [Swin Transformer (base version)](https://huggingface.co/microsoft/swin-base-patch4-window7-224)
- `swin_large` [Swin Transformer (large version)](https://huggingface.co/microsoft/swinv2-large-patch4-window12to16-192to256-22kto1k-ft)

---

### Custom models and datasets

If you are planning on using a new architecture, add it in the custom_models.py file and update the list in util.py. For new datasets, also add the name in the list if you want your models to be evaluated on it, ensure that they follow the same standards as mentionned in this readme, otherwise you will have to adapt the code to ensure a smooth running of the training/evaluation process.

---
## Acknowledgements
For the [dataset](https://www.epfl.ch/labs/mmspg/downloads/ai-synthesized-human-face-dataset/) and the directives during the project: [Yuhang Lu](https://scholar.google.com/citations?user=CtNglVsAAAAJ&hl=en).
<br>For the code: the [github](https://github.com/peterwang512/CNNDetection) of Peter Wang.
