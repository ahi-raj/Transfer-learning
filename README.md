# Transfer learning in image classification
## Mobilenet v2 pre-trained model from google's Tensorflow Hub and re-train that on flowers dataset for classification

```python
classifier = tf.keras.Sequential([
    hub.KerasLayer("https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/4", input_shape=IMAGE_SHAPE+(3,))
])
```
Make a classification prediction using a zebra image
![zebra](https://user-images.githubusercontent.com/94126896/174533564-decc7846-136d-4ab5-abff-ffdad831a20e.jpg =250x250)

<img src="[https://image-url.type](https://user-images.githubusercontent.com/94126896/174533564-decc7846-136d-4ab5-abff-ffdad831a20e.jpg)" width="200" height="200">

[[https://image-url.type](https://user-images.githubusercontent.com/94126896/174533564-decc7846-136d-4ab5-abff-ffdad831a20e.jpg|width=400px]]
[[link|width=400px]]

<img src="(https://your-image-url.type](https://github.com/ahi-raj/Transfer-learning/blob/main/zebra.jpg)" width="50%" height="50%">

<img src="https://user-images.githubusercontent.com/94126896/174535366-f614d5db-1232-43e1-a6be-d2ca30897c7c.jpg" data-canonical-src="https://user-images.githubusercontent.com/94126896/174535366-f614d5db-1232-43e1-a6be-d2ca30897c7c.jpg" width="200" height="200" />

![zebra](https://user-images.githubusercontent.com/94126896/174535366-f614d5db-1232-43e1-a6be-d2ca30897c7c.jpg)

<img src="https://user-images.githubusercontent.com/94126896/174535366-f614d5db-1232-43e1-a6be-d2ca30897c7c.jpg" width="50%" height="50%"/>
