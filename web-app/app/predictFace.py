import os
from app import APP_ROOT
import tensorflow as tf
from app.facealign import getRotated
import numpy as np

def predict_img():
  model = tf.keras.models.load_model(os.path.join(APP_ROOT, 'FP_model.h5'))
  img = getRotated(os.path.join(APP_ROOT, 'temp/'), 'face.jpg')
  os.remove(os.path.join(APP_ROOT, 'temp/face.jpg'))
  if img.shape[0] == 0: return -1
  pred = model.predict(np.array([img]))
  print(pred)
  return 1 if pred[0][0] > 0.5 else 0