# Convert the UFF to Nvidia's .plan (aka .engine)

The .plan and .engine files are needed to run the model on tensorrt. From my python-utilities
you may convert the keras model to frozen graph. Which in turn can be converted to nvidia's uff.
Using the c++ program in this folder, convert the uff to engine. Please be aware, the engine file
need to be generated on the machine-type on which you wish to run the tensorrt model for inference.
For example, if you wish to run inference on tx2, you need to generate the .plan (or .engine) file
on tx2. However, .UFF can still be generated on x86.

Borrowed from : https://github.com/NVIDIA-AI-IOT/tf_to_trt_image_classification/blob/master/src/uff_to_plan.cpp

## Compile
```
g++ -std=c++11 uff_to_plan.cpp -o uff_to_plan
    -I /usr/local/cuda/include -I /usr/local/TensorRT-5.1.5.0/include/
    -L /usr/local/cuda/lib64 -L /usr/local/TensorRT-5.1.5.0/targets/x86_64-linux-gnu/lib/  
    -lcudart -lnvinfer -lnvparsers
```

If you come across errors like NvInfer.h file not found, make sure the tensorrt headers are in your -I path.

If you come across linker errors like cannot find -lnvinfer, make sure your -L path have the libnvinfer.so and libnvparsers.so


## Execute
```
Usage: <uff_filename> <plan_filename> <input_name> <input_height> <input_width> <output_name> <max_batch_size> <max_workspace_size> <data_type>
```

My usage :
```
./uff_to_plan <your .uff file> ./kuse.plan input_1 480 640 net_vlad_layer_1/l2_normalize_1  1 0 float
```
