# Transfer learning in image classification
## Mobilenet v2 pre-trained model from google's Tensorflow Hub and re-train that on flowers dataset for classification

```python
classifier = tf.keras.Sequential([
    hub.KerasLayer("https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/4", input_shape=IMAGE_SHAPE+(3,))
])
```
Make a classification prediction using a zebra image
![alt text](http://url/to/img.png)