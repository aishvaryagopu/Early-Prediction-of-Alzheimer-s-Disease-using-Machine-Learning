from flask import *  
import os
import imageio
import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


from werkzeug.utils import secure_filename
app = Flask(__name__)  
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

class_label=""
 
@app.route('/upload',methods=['GET', 'POST'])  
def upload():  
    return render_template("Page.html")  
 
@app.route('/success', methods = ['POST'])  
def success():  
    if request.method == 'POST':  
        f = request.files['file']  
        f.save(f.filename)
        data = nib.load(f.filename)
        image_data = data.get_fdata()
        print(image_data.shape)
              
        image_name=r"static/coronal.png"
        data=image_data[:,135,:]
        data=np.rot90(data, 1)
        imageio.imwrite(image_name, data)
        image = cv2.imread(image_name, 1)
        bigger = cv2.resize(image, (227, 227))
        
        image_name="static/pred/pre/coronal.png"
        image_name1="static/coronal.png"
        data=bigger
        imageio.imwrite(image_name, data)
        imageio.imwrite(image_name1, data)
        
        image_name=r"static/sagittal.png"
        data=image_data[120,:,:]
        data=np.rot90(data, 1)
        imageio.imwrite(image_name, data)
        image = cv2.imread(r"static/sagittal.png", 1)
        bigger = cv2.resize(image, (227, 227))
      
        image_name="static/sagittal.png"
        data=bigger
        imageio.imwrite(image_name, data)
        
        image_name=r"static/axial.png"
        data=image_data[:,:,120]
        data=np.rot90(data, 1)
        imageio.imwrite(image_name, data)
        image = cv2.imread(r"static/axial.png", 1)
        bigger = cv2.resize(image, (227, 227))
      
        image_name="static/axial.png"
        data=bigger
        imageio.imwrite(image_name, data)

        return render_template("success.html", name = f.filename)

@app.route('/predict', methods = ['POST'])  
def predict():
    global class_label
    

    new_model=tf.keras.models.load_model('saved_model_alex/')
    path_test = 'static\\pred\\'
    
    predict_datagen = ImageDataGenerator(rescale=1. / 255)
    predict = predict_datagen.flow_from_directory(path_test, target_size=(227,227), batch_size = 1,class_mode='categorical')
    predictions = new_model.predict(predict)

    print(predictions[0])

    AD=predictions[0][0]*100
    CN=predictions[0][1]*100
    MCI=predictions[0][2]*100

    category=max(AD,CN,MCI)
    print(category)

    if(category==AD):
        class_label='Alzheimer Disease'
    elif(category==CN):
        class_label='Cognitively Normal'
    else:
        class_label='Mild Cognitive Impairment'
        
    print(class_label)

    return render_template("index.html", category_label=class_label,category_probability=round(category,2))  

@app.route('/diet', methods = ['POST'])  
def diet():
    global class_label
    if(class_label=='Alzheimer Disease'):
        return render_template("AD_diet.html") 
    if(class_label=='Cognitively Normal'):
        return render_template("avoid_AD.html")
    if(class_label=='Mild Cognitive Impairment'):
        return render_template("avoid_AD.html")  

@app.route('/braingame', methods = ['POST'])  
def braingame():
    return render_template("Games.html")  
    
    
if __name__ == '__main__':  
    app.run(debug = False)  
