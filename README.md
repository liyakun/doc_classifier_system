### Documents Classification Using Fine-Tuned Alexnet DNN

#### Components

##### Model Training
The `model_learn` folder contains the necessary tools to train a document classifier.

A trained deep neural network (dnn) machine learning model was trained to classify documents
The model was trained by fine tuning available train AlexNet dnn model.

###### Steps to reproduce
* download images
* download pre-trained AlexNet model
* prepare data using jupyter-notebook `model_learn/confs/utils/data_result_analysis.ipynb`
* retrain model using `model_learn/confs/pixel_mean_sub_shuffle/train.sh`

You could use the provided `model_learn/downloader.sh` to download images and model.

##### UI
The `ui` folder contains code for ui component.

The frontend ui provides the following functionality:
* upload image
* call backend to return classification and ocr results

##### OCR

The `ocr` folder contains code for ocr component.

The POST api provides the following functionality:
* receive image from `classifier`
* return ocr results to `classifier`

##### Classifier

The `classifier` folder contains code for classifier component.

The POST api provides the following functionality:
* receive image from `ui`
* return classification results to `ui`

#### Deployment
The deployment is done using docker-compose.

##### Docker Compose