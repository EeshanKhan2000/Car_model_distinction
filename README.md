# Car Model Distinction using STanford -196 Dataset.
## Aplied Localisation and then Classification for Better Performance

The MobileNetV2 Model architecture was used with Imagenet weights. The Top layers
were replaced with two densely connected layers of 1026 and 196 size respectively.
With dropout of 50% after penultimate layer. This was finetuned on the dataset.

However, in order to improve performance metrics, a cropping of ROI was done by applying
YOLO on the raw data, as part of the data pre-processing. By itself the classifier would not be
optimum as:
	1. Classes have extreme similarities. By allowing model to give more weightage to specific differences in cars would improove classification.
	2. Very small dataset (approx 100 images per class) would lead to poor training.
	3. Images contain background features (such as other cars) which would lead to weightage for irrelevant noise.


MobileNetV2 was preffered due to its light weight and speed.
This was also a personal exploration of Imageai, which makes using pre-trained models like YOLO
extremely simple, which would otherwise involve multiple steps.

Training was carried out for 50 epochs, after which validation accuracy was 83 %
![Training and Validation](https://github.com/EeshanKhan2000/Car_model_distinction/blob/master/Accuracy%26Loss.PNG)
