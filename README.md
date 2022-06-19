# Transfer learning in image classification
## Take pre-trained model from google's Tensorflow Hub and re-train that on flowers dataset

Make 
```python
classifier = tf.keras.Sequential([
    hub.KerasLayer("https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/4", input_shape=IMAGE_SHAPE+(3,))
])
```
