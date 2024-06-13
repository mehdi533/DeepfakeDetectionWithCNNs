# Exploring Diffusion-Generated Image Detection Methods

Research done by: Mehdi Abdallahi
<br>Supervisor: Yuhang Lu
<br>Professor: Dr. Touradj Ebrahimi

---

The content of this repository was used for the research done during the second semester of my Master's on the detection of diffusion-generated images, the code is adapted from the paper [CNN-generated images are surprisingly easy to spot... for now](https://arxiv.org/pdf/1912.11035.pdf) published by Wang et al. in 2020, and the code they used can be found in this [repository](https://github.com/peterwang512/CNNDetection). The [report]() and [presentation]() of this project are available.

---

For this research, the main focus was to gather useful insights for the generalization of entire face synthesis detection. To address this task, a state-of-the-art synthesized image detection method was first established as a baseline. In order to train and assess the performances of a wide range of models, a convenient pipeline for training, validation and evaluation was implemented. This allowed to move onto the next step: the exploration of large pre-trained vision models. More than 15 CNN architectures were explored, thanks to which interesting insights on the generalization of entire face synthesis classification were gathered.

---

### Getting started

```python
git clone https://github.com/mehdi533/XXXXXXX
cd XXXXXXXXX
```
The required libraries are in the requirements.txt file and can be installed using the command
```python
pip install -r requirements.txt
```

### Datasets

The datasets in the same format used for this study are available on the [MMSPG page](https://www.epfl.ch/labs/mmspg/downloads/ai-synthesized-human-face-dataset/).

The images of a specific generator (e.g. ProGAN) are all located in the same folder, the correct images for training, validation, and testing, are selected by loading the lists of images in the metadata folder. The emplacement of the Metadata folder has to be passed by argument, the default sets it in generated-image-detection/data.

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


If your dataset has the structure below:
```
├── data
│   ├── ProGAN
│   ├── PNDM
│   ├── CelebA
│   ├── ...

```

You can run this command in order to create the required file structure:
```
python data_restruct.py
```
Which automatically separates the data in train/val/test (0.8/0.1/0.1) sets. However, there is a slight issue with the restructuring and the `0_real` file does not appear properly in the test set, this can be fixed by moving the folder `data/test/0_real` into each dataset appearing under test i.e. `data/test/${dataset}/`.

> The datasets that will serve as training and test set need to be specified in the code, as well as the real dataset which will populate the `0_real` folder. `source_dir` and `dest_dir` variables need to be defined to define the location of the files.

---


---

### Training

In order to train your models you simply have to run, one command, which selects the augmentations, pre-processings, frequency transformations and datasets to run. The command with the settings that achieved our two best results, were:
```
# FFT
python train.py --name progan_pndm_bj_fft --fft --blur_prob 0.5 --blur_sig 0.0,3.0 --jpg_prob 0.5 --jpg_method cv2,pil --jpg_qual 30,100 --dataroot ./dataset/

# FFT + Low-Pass
python train.py --name progan_pndm_bj_fft_lp --low_pass --fft --blur_prob 0.5 --blur_sig 0.0,3.0 --jpg_prob 0.5 --jpg_method cv2,pil --jpg_qual 30,100 --dataroot ./dataset/
```
The possible flag options we propose are:
- `--fft`: Fast Fourier Transform (transform)
- `--dct`: Discrete Cosine Transform (transform)
- `--high_pass`: High Pass filter (pre-processing)
- `--thresh`: Thresholding (to be selected only if `--high_pass` is selected)
- `--low_pass`: Low Pass filter (pre-processing)
- `--edge`: Edge Detection (pre-processing)
- `--sharp_edge`: Sharp Edge Detection (pre-processing)

> The definition of these transformations/pre-processings can be found in `data/datasets.py`.

*Note: that only one transformation can be used at once (pre-processings could be used together)*

The model resulting from the training will be stored in the `/checkpoints` folder. 

---

### Evaluation 

To evaluate a model, the file named `eval_config.py` has to be modified. The `dataroot` variable should contain the path to the test folder, the `vals` variable should be a list containing the names of the datasets to be evaluated (subdirectories of test) and the `model_path` variable should contain the path to the weights of the model that has to be evaluated (`*.pth` file).

When these are configured, we can start the evaluation. We can run the command below:
```
# FFT
python eval.py --fft

# FFT + Low-Pass
python eval.py --fft --low_pass
```
_**Important:** make sure that the pre-processing and the transformations of the evaluation are the same as the ones used to train the concerned model (don't include the augmentations)._

The results of the evaluation will appear in the cmd as well as in the `results` folder.

You can download the models proposed by Wang et al. and evaluate them on your datasets for a quick test. The models can be downloaded running this command:
```
bash weights/download_weights.sh
```

---
### Frequency Heatmaps and Spectra

The frequency heatmaps and spectra that can be found in the report and the presentation can be obtained by running the scripts in the `heatmap_spectra_generation` folder. To be able to run this with your own datasets, you will need to add your dataset in the `datasets` dictionary and also inlude it in the argument parser in the possible choices of datasets.

To create a frequency heatmap the only required argument is the dataset to be analyzed (for the high-pass script there is the possibility to use a threshold or not, the default option is a threshold of 1).

```
python heatmap_spectra_generation/hp_heatmap.py --dataset progan --no_thresh
```

Example of call:

```python
python train.py --arch swin_tiny --name 1405_print --no-intermediate --batch_size 256 --models real,ProGAN,DDIM
```




Example of call:

```python
python train.py --arch swin_tiny --name 1405_print --no-intermediate --batch_size 256 --models real,ProGAN,DDIM
```
