# Who am I
Dylan Taft, professionally I am an IT worker who has been in a lot of different positions over the years, support roles, AIX sysadmin, systems engineer for Microsoft operating systems, SQL based analytics development.  Privately I am a tinkerer of sorts and am interested in training AI models for personal projects.

# What is this
I wrote a quick test script to evaluate usage and performance of the Ampere Optimized Onnx runtime vs the vanilla upstream Onnx runtime with yolo models
https://hub.docker.com/r/amperecomputingai/onnxruntime
https://amperecomputing.com/assets/Ampere_Optimized_ONNXRuntime_Documentation_v1_8_0_9646259707.pdf

This repo contains the script I tested, and this document contains my opinions and findings as a hobbiest, not a professional in this arena.

# Prereqs
Install the docker image documented here 
https://hub.docker.com/r/amperecomputingai/onnxruntime
https://amperecomputing.com/assets/Ampere_Optimized_ONNXRuntime_Documentation_v1_8_0_9646259707.pdf

This was evaluated against Ultralytics World model - you will need to download this
https://docs.ultralytics.com/models/yolo-world/ 
YOLOv8s-worldv2

Outside of the docker image, download and install Yolo per their instructions - pip pathway is easy enough
https://docs.ultralytics.com/quickstart/

You would export the .pt pytorch model to onnx with the following commmand
yolo export model=yolov8s-worldv2.pt format=onnx opset=12
This model is copywritten by Ultralytics and can't be included in this repo, to my knowledge, but it is provided by them on their website.

# Usage
Copy the model and the script to the docker image
python test.py - it will download coco128 image set from ultralytics, letterbox to 640x640, and run inference according to configuration in the .py file

# Things of interest to set
sess_options.intra_op_num_threads = 32
sess_options.inter_op_num_threads = 1
outer_threads = 1

# General Findings
The docker image provides a quick way to be up and running on an optimized version of the Onnx runtime for Ampere.  Out of the box, for the yolov8s world model, it appears to be roughly 80% faster at single, unbatched detection inference vs the vanilla implementation.

# Positive findings
- It performs much faster at single image inferance - 80% in my test case on q22-64, yolo v8 world v2.
- It is quick to install

# Negative findings
- The docker image at the time of this writing is a year old, a lot changes have been upstreamed in Onnx since then.  Recent versions of Onnx and Yolo pose compatibility issues, with exporting of the yolo model requiring that opset parameter
- As far as I can tell, the C and C++ bindings as well as shared library are not available in the docker image, eschewing the ability to use the optimized framework in languages other than python
- While the threading is fast - it is typical, possible, maybe recommended to batch or thread inference runs in parallel - especially for large images.  You can break up a 1920x1080 or 4k image into 640x640 windows if you need to run object detection against large images.  There are other ways to do this as well, resizing tensors down to the model's expected format, or training a model with a larger input tensor.  There's dynamic input sized tensors, but generally using the expected tensor size works the best.  Setting outer_threads and running multiple copies of inference in paralle, the performance gains are erased.

# Things that would be nice to have
- C + C++ bindings, headers, shared library
- Frequent updates
- Having this outside of Docker - onnx allows for third party "providers".   The ability to build or link Onnx runtime from source with Ampere's provider, even in binary format, would be nice.

# Caveats
I have put this together quickly as my personal project is in C, so I cannot use the docker image which only supports python.  The performance is about what I expect, but the example is not complete - no image output, I have not validated that objects are being detected, but I have in my personal C project, and anticipate if this benchmark was fully fleshed out, it would be fine.

# Final Thoughts
Works fine.  If you need to quickly get up on OnnxRuntime on Ampere, and you are doing single batch inference, go for it.  Have not tested multibatch on anything but Vanilla.  






