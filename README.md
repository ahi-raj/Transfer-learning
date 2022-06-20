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
From: https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos
