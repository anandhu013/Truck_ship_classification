from keras.models import load_model
from keras.preprocessing import image
import numpy as np

model = load_model('model.h5') 
def predict(img,model=model):
	x = image.img_to_array(img)
	x = np.true_divide(x, 255)
	x = np.expand_dims(x, axis=0)
	preds = model.predict(x)
	return preds