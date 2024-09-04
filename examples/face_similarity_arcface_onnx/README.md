# tract onnx face similarity comparison using arcface
face deteciton demo using yolov8 , using models converted from [derronqi's repo](https://github.com/derronqi/yolov8-face), and then using ArcFace to get face embeddings and then compare them with cosine similarity.

# getting the models
## yolov8-face
1. you can get the model and convert them yourself from [here](https://github.com/derronqi/yolov8-face) , you can follow conversion instructions [here](https://docs.ultralytics.com/integrations/onnx/)
2. you can get it preconverted from
[google drive](https://drive.google.com/file/d/1PYAG1ypAuwh_rDROaUF0OdLmBqOefBGL/view?usp=sharing)

## arcface 
you can get onnx converted models [here](https://github.com/onnx/models/tree/main/validated/vision/body_analysis/arcface)


# to use 
run `cargo run -- --face1 path/to/image1 --face2 path/to/image2`
