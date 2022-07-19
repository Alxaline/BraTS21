# A complete pipeline for BraTS 2021: [RSNA-ASNR-MICCAI Brain Tumor Segmentation (BraTS) Challenge 2021 ](https://www.med.upenn.edu/cbica/brats2021/)

This repository implement the solution of the 2021 edition of the BraTS challenge describe in our [paper][brats2021]. 

If you face any problem, please feel free to open an issue.

## Installation

###  1. Get the repository
```
git clone https://github.com/Alxaline/BraTS21.git
cd BraTS21
```

### 2. Create a [conda](https://docs.conda.io/en/latest/) environment (recommended)
```
ENVNAME="BraTS21"
conda create -n $ENVNAME python==3.7.7 -y
conda activate $ENVNAME
```

### 3. Install requirements

in your favorite virtual environment:

```
pip install -r requirements.txt
```

## Data directory structure

The hierarchical structure of the data folder should be as follows:

	├─ data
	  ├─ BraTS2021	# Data provided by the BraTS 2020 competition host
	    ├─ RSNA_ASNR_MICCAI_BraTS2021_TrainingData
		    ├─ BraTS2021_00000
		        ├─ BraTS2021_00000_flair.nii.gz
		        ├─ BraTS2021_00000_seg.nii.gz
		        ├─ BraTS2021_00000_t1.nii.gz
		        ├─ BraTS2021_00000_t1ce.nii.gz
		        ├─ BraTS2021_00000_t2.nii.gz
			├─ BraTS2021_00002
			├─ ...
	    ├─ RSNA_ASNR_MICCAI_BraTS2021_ValidationData
			├─ BraTS20_Validation_001
		    	├─ BraTS20_Validation_001_flair.nii.gz
		    	├─ BraTS20_Validation_001_t1.nii.gz
		    	├─ BraTS20_Validation_001_t1ce.nii.gz
		    	├─ BraTS20_Validation_001_t2.nii.gz
			├─ BraTS2021_00013
		    ├─ ...

## Results of the paper

### On validation

| **Model Nb.** | **Sub. ID** | **Method**                                      |  **DSC WT**  |  **DSC TC**  |  **DSC ET**  | **DSC Mean** | **HD95 WT**  | **HD95 TC**  |  **HD95 ET**  | **HD95 Mean** |
|---------------|:-----------:|:------------------------------------------------|:------------:|:------------:|:------------:|:------------:|:------------:|:------------:|:-------------:|:--------------|
| 1)            |   9715210   | U-Net<sub>V1</sub><sup>(\*)</sup>               |   0\.91904   |   0\.86616   |   0\.83454   |   0\.87326   |   4\.40718   |   9\.39596   |   15\.75011   | 9\.85108      |
| 2)            |   9715055   | U-Net<sub>V2</sub><sup>(\*)</sup>               |   0\.92349   |   0\.86827   |   0\.83265   |   0\.87475   | **4\.12874** |  10\.92845   |   17\.48075   | 10\.84598     |
| 3)            |   9715112   | U-Net<sub>V2</sub><sup>(\*\*)</sup>             |   0\.92393   |   0\.87063   |   0\.83997   |   0\.87782   |   4\.61502   |   9\.34665   |   15\.80434   | 9\.92200      |
| 4)            |   9715113   | U-Net<sub>V2</sub><sup>(\*+\*\*)</sup>          |   0\.92436   |   0\.87168   |   0\.84000   |   0\.87868   |   4\.49349   |   7\.71372   |   14\.15743   | 8\.78821      |
| 5)            |   9715160   | U-Net<sub>V2</sub><sup>(\*, JL)</sup>           |   0\.92462   |   0\.87712   |   0\.83994   |   0\.88056   |   4\.25690   |   9\.21011   |   14\.16697   | 9\.21133      |
| 6)            |   9715224   | U-Net<sub>V2</sub><sup>(\*\*+(\*, JL))</sup>    |   0\.92457   | **0\.87811** | **0\.84094** | **0\.88121** |   4\.19442   |   7\.55256   | **14\.13390** | **8\.62696**  |
| 7)            |   9715209   | U-Net<sub>V2</sub><sup>(\*+\*\*+(\*, JL))</sup> | **0\.92463** |   0\.87674   |   0\.83916   |   0\.88018   |   4\.48539   | **7\.53955** |   15\.75771   | 9\.26088      |

### On test

Model 6 was the model selected for final test. Results provided by BraTS organization were as follows:

|            | **DSC WT** | **DSC TC** | **DSC ET** | **HD95 WT** | **HD95 TC** | **HD95 ET** |
|:-----------|-----------:|-----------:|-----------:|------------:|------------:|------------:|
| Mean       |   0\.92548 |   0\.87628 |   0\.87122 |    4\.30711 |   17\.84987 |   12\.23361 |
| StdDev     |   0\.09898 |   0\.23983 |   0\.18204 |    8\.45388 |   71\.52831 |   59\.54562 |
| Median     |   0\.95560 |   0\.95975 |   0\.93153 |    1\.73205 |    1\.41421 |    1\.00000 |
| 25quantile |   0\.91378 |   0\.91195 |   0\.85409 |    1\.00000 |    1\.00000 |    1\.00000 |
| 75quantile |   0\.97604 |   0\.97916 |   0\.95977 |    4\.00000 |    3\.00000 |    2\.00000 |

## Training

For more details on the available options:
```
python -m src.main_train -h
```

In order to perform the different models proposed in the paper, run the following commands.

After a run, in the `--save_path` args, a folder will be created containing :
- a 'config.yaml' file with the option used 
- model weights (best and last)
- a 'Evaluation.xlsx' file with metrics result on the evaluation set
- a 'segmentations' folder containing all the generated seg on the evaluation set

with the --evaluate_end_training args, best model weight will be run on the evaluation set

For each model, a cross validation was performed. 
For this each command must be run again with a different `--fold` arg (0, 1, 2, 3 or 4) 

### 1) U-Net<sub>V1</sub><sup>(\*)</sup>

```
python -m src.main_train --train_data_path /data/RSNA_ASNR_MICCAI_BraTS2021_TrainingData/ --save_path /data/model1/fold0 --model equiunet --norm group --act relu --width 48 --criterion dice --num_workers 4 --optimizer ranger --decay_type cosine --learning_rate 0.0003 --val_frequency 2 --log_val_metrics --evaluate_end_training --remove_outliers --epochs 150  --no_full_name --fold 0 --device 0 -vv
```

### 2) U-Net<sub>V2</sub><sup>(\*)</sup>

```
python -m src.main_train --train_data_path /data/RSNA_ASNR_MICCAI_BraTS2021_TrainingData/ --save_path /data/model2/fold0 --model equiunet_assp_evo --act leakyrelu --width 48 --criterion dice --num_workers 4 --optimizer ranger --decay_type cosine --learning_rate 0.0003 --val_frequency 2 --log_val_metrics --evaluate_end_training --remove_outliers --epochs 150 --no_full_name --fold 0 --device 0 -vv
```

### 3) U-Net<sub>V2</sub><sup>(\*\*)</sup> 

```
python -m src.main_train --train_data_path /data/RSNA_ASNR_MICCAI_BraTS2021_TrainingData/ --save_path /data/model3/fold0 --model equiunet_assp_evo --act leakyrelu --width 48 --criterion dice --num_workers 4 --optimizer ranger --decay_type cosine --learning_rate 0.0003 --val_frequency 2 --log_val_metrics --evaluate_end_training --remove_outliers --epochs 150 --no_full_name --fold 0 --device 0 --seed 93 -vv
```

### 5) U-Net<sub>V2</sub><sup>(\*, JL)</sup>

```
python -m src.main_train --train_data_path /data/RSNA_ASNR_MICCAI_BraTS2021_TrainingData/ --save_path /data/model5/fold0 --model equiunet_assp_evo --act leakyrelu --width 48 --criterion jaccard --num_workers 4 --optimizer ranger --decay_type cosine --learning_rate 0.0003 --val_frequency 2 --log_val_metrics --evaluate_end_training --remove_outliers --epochs 150 --no_full_name --fold 0 --device 0 -vv
```

## Inference

For more details on the available options:
```
python -m src.main_inference -h
```

### 1) U-Net<sub>V1</sub><sup>(\*)</sup>

```
python -m src.main_inference --config /data/model1/fold0/config.yaml /data/model1/fold1/config.yaml /data/model1/fold2/config.yaml /data/model1/fold3/config.yaml /data/model1/fold4/config.yaml --test_data_path /data/RSNA_ASNR_MICCAI_BraTS2021_ValidationData --on test -vv --replace_value --cleaning_areas --save_path /data/model1_inference/ --device 0 --replace_value_threshold 300 --cleaning_areas_threshold 20 --device 0 --tta
```

### 2) U-Net<sub>V2</sub><sup>(\*)</sup>

```
python -m src.main_inference --config /data/model2/fold0/config.yaml /data/model2/fold1/config.yaml /data/model2/fold2/config.yaml /data/model2/fold3/config.yaml /data/model2/fold4/config.yaml --test_data_path /data/RSNA_ASNR_MICCAI_BraTS2021_ValidationData --on test -vv --replace_value --cleaning_areas --save_path /data/model2_inference/ --device 0 --replace_value_threshold 300 --cleaning_areas_threshold 20 --device 0 --tta
```

### 3) U-Net<sub>V2</sub><sup>(\*\*)</sup> 

```
python -m src.main_inference --config /data/model3/fold0/config.yaml /data/model3/fold1/config.yaml /data/model3/fold2/config.yaml /data/model3/fold3/config.yaml /data/model3/fold4/config.yaml --test_data_path /data/RSNA_ASNR_MICCAI_BraTS2021_ValidationData --on test -vv --replace_value --cleaning_areas --save_path /data/model3_inference/ --device 0 --replace_value_threshold 300 --cleaning_areas_threshold 20 --device 0 --tta
```

### 4) U-Net<sub>V2</sub><sup>(\*+\*\*)</sup>

```
python -m src.main_inference --config /data/model1/fold0/config.yaml /data/model1/fold1/config.yaml /data/model1/fold2/config.yaml /data/model1/fold3/config.yaml /data/model1/fold4/config.yaml /data/model3/fold0/config.yaml /data/model3/fold1/config.yaml /data/model3/fold2/config.yaml /data/model3/fold3/config.yaml /data/model3/fold4/config.yaml --test_data_path /data/RSNA_ASNR_MICCAI_BraTS2021_ValidationData --on test -vv --replace_value --cleaning_areas --save_path /data/model4_inference/ --device 0 --replace_value_threshold 300 --cleaning_areas_threshold 20 --device 0 --tta
```

### 5) U-Net<sub>V2</sub><sup>(\*, JL)</sup>

```
python -m src.main_inference --config /data/model5/fold0/config.yaml /data/model5/fold1/config.yaml /data/model5/fold2/config.yaml /data/model5/fold3/config.yaml /data/model5/fold4/config.yaml --test_data_path /data/RSNA_ASNR_MICCAI_BraTS2021_ValidationData --on test -vv --replace_value --cleaning_areas --save_path /data/model5_inference/ --device 0 --replace_value_threshold 300 --cleaning_areas_threshold 20 --device 0 --tta
```

### 6) U-Net<sub>V2</sub><sup>(\*\*+(\*, JL))</sup> 

```
python -m src.main_inference --config /data/model3/fold0/config.yaml /data/model3/fold1/config.yaml /data/model3/fold2/config.yaml /data/model3/fold3/config.yaml /data/model3/fold4/config.yaml /data/model5/fold0/config.yaml /data/model5/fold1/config.yaml /data/model5/fold2/config.yaml /data/model5/fold3/config.yaml /data/model5/fold4/config.yaml --test_data_path /data/RSNA_ASNR_MICCAI_BraTS2021_ValidationData --on test -vv --replace_value --cleaning_areas --save_path /data/model6_inference/ --device 0 --replace_value_threshold 300 --cleaning_areas_threshold 20 --device 0 --tta
```

### 7) U-Net<sub>V2</sub><sup>(\*+\*\*+(\*, JL))</sup>

```
python -m src.main_inference --config /data/model2/fold0/config.yaml /data/model2/fold1/config.yaml /data/model2/fold2/config.yaml /data/model2/fold3/config.yaml /data/model2/fold4/config.yaml /data/model3/fold0/config.yaml /data/model3/fold1/config.yaml /data/model3/fold2/config.yaml /data/model3/fold3/config.yaml /data/model3/fold4/config.yaml /data/model5/fold0/config.yaml /data/model5/fold1/config.yaml /data/model5/fold2/config.yaml /data/model5/fold3/config.yaml /data/model5/fold4/config.yaml --test_data_path /data/RSNA_ASNR_MICCAI_BraTS2021_ValidationData --on test -vv --replace_value --cleaning_areas --save_path /data/model6_inference/ --device 0 --replace_value_threshold 300 --cleaning_areas_threshold 20 --device 0 --tta
```

### Extra : Final weights used for test 

The weights of our model for the BraTS 2021 challenge can be downloaded at
https://drive.google.com/file/d/1Xt2rdD60IeEwcd8-yiMZHZkI0udcXgc7/view?usp=sharing

Unzip the `final_weights_brats21.zip` and put the folder in `BraTS21`.
Execute the following command line and replace the path of `--input` and `output args`

```
python -m src.main_inference --config /BraTS21/final_weights_brats21/baseline_equiunet_assp_evocor/fold0_ns/config.yaml /BraTS21/final_weights_brats21/baseline_equiunet_assp_evocor/fold1_ns/config.yaml /BraTS21/final_weights_brats21/baseline_equiunet_assp_evocor/fold2_ns/config.yaml /BraTS21/final_weights_brats21/baseline_equiunet_assp_evocor/fold3_ns/config.yaml /BraTS21/final_weights_brats21/baseline_equiunet_assp_evocor/fold4_ns/config.yaml /BraTS21/final_weights_brats21/baseline_equiunet_assp_evocor_jaccard/fold0/config.yaml /BraTS21/final_weights_brats21/baseline_equiunet_assp_evocor_jaccard/fold1/config.yaml /BraTS21/final_weights_brats21/baseline_equiunet_assp_evocor_jaccard/fold2/config.yaml /BraTS21/final_weights_brats21/baseline_equiunet_assp_evocor_jaccard/fold3/config.yaml /BraTS21/final_weights/baseline_equiunet_assp_evocor_jaccard/fold4/config.yaml --on test -vv --replace_value --cleaning_areas --replace_value_threshold 300 --cleaning_areas_threshold 20 --device 0 --tta --input /input --output /output --num_workers 0
```

## Docker

Retrieve the docker image of the final model used for test :

```
docker pull alxaline/brats21:latest
```

To run on a test sample : 

```
sudo docker run -it --rm --gpus device=0 --name run_model -v "/data/RSNA_ASNR_MICCAI_BraTS2021_TestingData/BraTS20_Testing_006":"/input" -v "/data/outputseg/":"/output" alxaline/brats21 --input /input --output /output
```



## How to cite

If you find this repository useful for your research, please cite our work
* Carré, A., Deutsch, E., Robert, C. (2022). [Automatic Brain Tumor Segmentation with a Bridge-Unet Deeply Supervised Enhanced with Downsampling Pooling Combination, Atrous Spatial Pyramid Pooling, Squeeze-and-Excitation and EvoNorm.][brats2021] In: Crimi, A., Bakas, S. (eds) Brainlesion: Glioma, Multiple Sclerosis, Stroke and Traumatic Brain Injuries. BrainLes 2021. Lecture Notes in Computer Science, vol 12963. Springer, Cham. https://doi.org/10.1007/978-3-031-09002-8_23
* Henry, T. et al. (2021). [Brain Tumor Segmentation with Self-ensembled, Deeply-Supervised 3D U-Net Neural Networks: A BraTS 2020 Challenge Solution.][brats2020] In: Crimi, A., Bakas, S. (eds) Brainlesion: Glioma, Multiple Sclerosis, Stroke and Traumatic Brain Injuries. BrainLes 2020. Lecture Notes in Computer Science(), vol 12658. Springer, Cham. https://doi.org/10.1007/978-3-030-72084-1_30

BibTeX:
```
@inproceedings{carreAutomaticBrainTumor2022,
	location = {Cham},
	title = {Automatic Brain Tumor Segmentation with a Bridge-Unet Deeply Supervised Enhanced with Downsampling Pooling Combination, Atrous Spatial Pyramid Pooling, Squeeze-and-Excitation and {EvoNorm}},
	isbn = {978-3-031-09002-8},
	doi = {10.1007/978-3-031-09002-8_23},
	series = {Lecture Notes in Computer Science},
	pages = {253--266},
	booktitle = {Brainlesion: Glioma, Multiple Sclerosis, Stroke and Traumatic Brain Injuries},
	publisher = {Springer International Publishing},
	author = {Carré, Alexandre and Deutsch, Eric and Robert, Charlotte},
	editor = {Crimi, Alessandro and Bakas, Spyridon},
	year = {2022},
	language = {en},
	keywords = {Brain tumor, Deep-learning, Segmentation},
}
@inproceedings{henryBrain2021,
	location = {Cham},
	title = {Brain Tumor Segmentation with Self-ensembled, Deeply-Supervised 3D U-Net Neural Networks: A BraTS 2020 Challenge Solution},
	isbn = {978-3-030-72084-1},
	doi = {10.1007/978-3-030-72084-1_30},
	series = {Lecture Notes in Computer Science},
	pages = {327--339},
	booktitle = {Brainlesion: Glioma, Multiple Sclerosis, Stroke and Traumatic Brain Injuries},
	publisher = {Springer International Publishing},
	author = {Henry, Théophraste and Carré, Alexandre and Lerousseau, Marvin and Estienne, Théo and Robert, Charlotte and Paragios, Nikos and Deutsch, Eric},
	editor = {Crimi, Alessandro and Bakas, Spyridon},
	year = {2021},
	language = {en},
	keywords = {Brain tumor, Deep learning, Semantic segmentation},
}
```

[brats2021]: https://link.springer.com/chapter/10.1007/978-3-031-09002-8_23
[brats2020]: https://link.springer.com/chapter/10.1007/978-3-030-72084-1_30
