rm ../models/labels.txt
python3 facenet_tpu.py --model ~/facenet/my_facenet2_1559545916_edgetpu.tflite --label-make True
python3 detect.py --model ~/facenet/mobilenet_ssd_v2_face_quant_postprocess_edgetpu.tflite
