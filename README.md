### Documents Classification Using Fine-Tuned Alexnet DNN

#### Components

##### Model Training
The `model_learn` folder contains the necessary tools to train a document classifier.

A deep neural network (dnn) machine learning model was trained to classify documents. The model was trained by fine tuning available trained AlexNet dnn model.

You could find training log file at `doc_classifier_system/model_learn/confs/pixel_mean_sub_shuffle/training.log`. 
The plotted confusion matrix could be found at `model_learn/confs/utils/data_result_analysis.ipynb`.


###### Steps to reproduce
* download images
* download pre-trained AlexNet model
* prepare data using jupyter-notebook `model_learn/confs/utils/data_result_analysis.ipynb`
  * `data_result_analysis.ipynb` under `model_learn/confs/utils/` is developed by me, other files under this folder are mainly taken from official caffe github repository.
* retrain model using `model_learn/confs/pixel_mean_sub_shuffle/train.sh`

The provided `model_learn/downloader.sh` could be used to download images and model.

##### UI
The `ui` folder contains code for ui component.

The frontend ui provides the following functionality:
* upload image
* callback to return classification and ocr results

##### OCR

The `ocr` folder contains code for ocr component.

The `ocr` component provides the following functionality:
* receive image from `classifier`
* return ocr results to `classifier`

##### Classifier

The `classifier` folder contains code for classifier component.

The `classifier` provides the following functionality:
* receive image from `ui`
* return classification results to `ui`

#### Deployment

The deployment is done through docker, therefore, it is required to have docker and docker-compose installed.
You could use the provided bash script to deploy the three components.

1. clone the repository
    ```bash
    git clone https://gitlab.com/yakun/doc_classifier_system.git
    cd ./doc_classifier_system && git checkout remotes/origin/rabbitmq
    ```

2. download trained model
    ```bash
    sh downloader.sh
    ```
3. run docker-compose.yml
    ```bash
    sh run.sh
    ```
4. access ui running at http://localhost:9999/ to upload image

#### References
1. https://www.rabbitmq.com/getstarted.html
2. https://github.com/NVIDIA/DIGITS
3. https://deshanadesai.github.io/notes/Finetuning-ImageNet
4. http://adilmoujahid.com/posts/2016/06/introduction-deep-learning-python-caffe/