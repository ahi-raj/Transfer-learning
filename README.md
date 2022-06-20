# Transfer learning in image classification
## Mobilenet v2 pre-trained model from google's Tensorflow Hub and re-train that on flowers dataset for classification

```python
classifier = tf.keras.Sequential([
    hub.KerasLayer("https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/4", input_shape=IMAGE_SHAPE+(3,))
])
```
Make a classification prediction using a zebra image

<img src="https://user-images.githubusercontent.com/94126896/174535366-f614d5db-1232-43e1-a6be-d2ca30897c7c.jpg" width="50%" height="50%"/>

## Load flowers dataset 
From: https://www.tensorflow.org/datasets/catalog/tf_flowers
```python
dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
data_dir = tf.keras.utils.get_file('flower_photos', origin=dataset_url,  cache_dir='.', untar=True)
# cache_dir indicates where to download data. I specified . which means current directory
# untar true will unzip it
```
Read flowers images from disk into numpy array using opencv

Preprocess and split the dataset for traning and testing

Retrain the flower images on pretrained mobilenet v2 model using feature vector

After compiling the model the training accuracy of 92% is achieved and testing accuracy of 86%
