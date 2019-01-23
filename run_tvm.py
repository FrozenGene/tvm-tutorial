from PIL import Image
import numpy as np

def preprocess_image(image_file):
    resized_image = Image.open(image_file).resize((224, 224))
    image_data = np.asarray(resized_image).astype("float32")
    # convert HWC to CHW
    image_data = image_data.transpose((2, 0, 1))
    # after expand_dims, we have format NCHW
    image_data = np.expand_dims(image_data, axis = 0)
    image_data[:,0,:,:] = 2.0 / 255.0 * image_data[:,0,:,:] - 1 
    image_data[:,1,:,:] = 2.0 / 255.0 * image_data[:,1,:,:] - 1
    image_data[:,2,:,:] = 2.0 / 255.0 * image_data[:,2,:,:] - 1
    return image_data

def run(model_file, image_data):
    import tflite.Model
    from tvm import relay

    # open TFLite model file
    buf = open(model_file, 'rb').read()

    # get TFLite model data structure
    tflite_model = tflite.Model.Model.GetRootAsModel(buf, 0)

    # TFLite input tensor name, shape and type
    input_tensor = "input"
    input_shape = (1, 3, 224, 224)
    input_dtype = "float32"

    # parse TFLite model and convert into Relay computation graph
    func, params = relay.frontend.from_tflite(tflite_model, shape_dict={input_tensor: input_shape}, dtype_dict={input_tensor: input_dtype})

    # targt x86 cpu
    target = "llvm"
    with relay.build_module.build_config(opt_level=3):
        graph, lib, params = relay.build(func, target, params=params)

    import tvm
    from tvm.contrib import graph_runtime as runtime
    # create a runtime executor module
    module = runtime.create(graph, lib, tvm.cpu())
    # feed input data
    module.set_input(input_tensor, tvm.nd.array(image_data))
    # feed related params
    module.set_input(**params)
    # run
    module.run()
    # get output
    tvm_output = module.get_output(0).asnumpy()
    
    return tvm_output

def post_process(tvm_output, label_file):
    # map id to 1001 classes
    labels = dict()
    with open(label_file) as f:
        for id, line in enumerate(f):
            labels[id] = line
    # convert result to 1D data
    predictions = np.squeeze(tvm_output)
    # get top 1 prediction
    prediction = np.argmax(predictions)

    # convert id to class name
    print("The image prediction result is: id " + str(prediction) + " name: " + labels[prediction])

if __name__ == "__main__":
    image_file = 'cat.png'
    model_file = 'mobilenet_v1_1.0_224.tflite'
    label_file = 'labels_mobilenet_quant_v1_224.txt'
    image_data = preprocess_image(image_file)
    tvm_output = run(model_file, image_data)
    post_process(tvm_output, label_file)

    import run_tflite
    tflite_image_data = run_tflite.preprocess_image(image_file)
    tflite_output = run_tflite.run(model_file, tflite_image_data)

    # compare tvm_output and tflite_output
    np.testing.assert_allclose(tvm_output, tflite_output, rtol=1e-5, atol=1e-5)
    print("Pass!")
