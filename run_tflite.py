from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.contrib.lite.python import interpreter as interpreter_wrapper

def preprocess_image(image_file):
    resized_image = Image.open(image_file).resize((224, 224))
    image_data = np.asarray(resized_image).astype("float32")
    image_data = np.expand_dims(image_data, axis = 0)
    image_data[:,:,:,0] = 2.0 / 255.0 * image_data[:,:,:,0] - 1 
    image_data[:,:,:,1] = 2.0 / 255.0 * image_data[:,:,:,1] - 1
    image_data[:,:,:,2] = 2.0 / 255.0 * image_data[:,:,:,2] - 1
    return image_data

def run(model_file, image_data):
    interpreter = interpreter_wrapper.Interpreter(model_path=model_file)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # set input
    interpreter.set_tensor(input_details[0]['index'], image_data)

    # Run
    interpreter.invoke()

    # get output
    tflite_output = interpreter.get_tensor(output_details[0]['index'])

    return tflite_output

def post_process(tflite_output, label_file):
    # map id to 1001 classes
    labels = dict()
    with open(label_file) as f:
        for id, line in enumerate(f):
            labels[id] = line
    # convert result to 1D data
    predictions = np.squeeze(tflite_output)
    # get top 1 prediction
    prediction = np.argmax(predictions)

    # convert id to class name
    labels = dict()
    with open(label_file) as f:
        for i, line in enumerate(f):
            labels[i] = line
    print("The image prediction result is: id " + str(prediction) + " name: " + labels[prediction])

if __name__ == "__main__":
    image_file = 'cat.png'
    model_file = 'mobilenet_v1_1.0_224.tflite'
    label_file = 'labels_mobilenet_quant_v1_224.txt'
    image_data = preprocess_image(image_file)
    tflite_output = run(model_file, image_data)
    post_process(tflite_output, label_file)