from edgetpu.classification.engine import ClassificationEngine
from edgetpu.detection.engine import DetectionEngine
from edgetpu.utils.image_processing import ResamplingWithOriginalRatio 
import numpy as np
from PIL import Image
import argparse
import math
from functools import reduce
from collections import OrderedDict
#import picamera, io
import gstreamer, svgwrite, imp
import time
import copy

from utils.facenet_custom import Utils


class FacenetEngine(ClassificationEngine):

    def __init__(self, model_path, device_path=None):
        if device_path:
            super().__init__(model_path, device_path)
        else:
            super().__init__(model_path)
    def generate_labelfile(self, data_folder="/home/mendel/facenet/lfw_mtcnnalign_160_ab/", save_as="/home/mendel/coral_utils/models/labels.txt", avg_only=True, per_class_img_num=5):
        """
        generate embedding vector label file with corresponding directory.
        If avg_only is False, write down every embedding vector as label file.
        """
        NORMALIZE = False
        import glob
        members = glob.glob(data_folder+"*")
        with open(save_as, "w") as write_file:
            output_shape = self.get_all_output_tensors_sizes()
            print("output shape" ,output_shape)
            for name_folder in members:
                images = glob.glob(name_folder+"/*")
                avg_ev = np.zeros(output_shape[0], ) #[128,1] or [512,1]
                i=0
                for img_path in images:
                    img = Image.open(img_path)
                    _, _, ev = self.GetEmbeddingVector(img)
                    print(img_path)
                    if avg_only is False:
                        write_file.write(name_folder.split("/")[-1] + " ")
                        for j, elem in enumerate(ev):
                            if j == len(avg_ev) - 1:
                                write_file.write(str(elem))
                            else:
                                write_file.write(str(elem)+ ",")
                        write_file.write("\n")
                    else:
                        avg_ev = avg_ev + ev
                    i += 1
                    if i == per_class_img_num:
                        print("break")
                        break
                if avg_only is True:
                    avg_ev = avg_ev / per_class_img_num
                    #normalize
                    if NORMALIZE:
                        avg_ev = avg_ev / np.linalg.norm(avg_ev)
                    write_file.write(name_folder.split("/")[-1] + " ")
                    for j, elem in enumerate(avg_ev):
                        if j == len(avg_ev) - 1:
                            write_file.write(str(elem))
                        else:
                            write_file.write(str(elem)+ ",")
                    write_file.write("\n")

    def ImportLabel(self, label_file="/home/mendel/coral_utils/models/labels.txt", knn=False):
        """
        jwpyo, [1, 2, ... ]
        If import_all is True, import all of face datas. IF not, import average of face datas.
        """

        self.label_dict = dict()
        self.label_id = dict()
        #for KNN
        self.data_factory = list()
        if knn is True:
            with open(label_file, "r") as lf:
                for line in lf:
                    class_name, raw_ev = line.split(" ")
                    ev = list(map(float, raw_ev.split(",")))
                    self.data_factory.append((ev, class_name))
        else:
            with open(label_file, "r") as lf:
                for line in lf:
                    class_name, raw_ev = line.split(" ")
                    ev = list(map(float, raw_ev.split(",")))
                    self.label_id[len(self.label_dict)] = class_name
                    self.label_dict[class_name] = ev
            #check if it is unit vector
            for i in self.label_id.keys():
                name = self.label_id[i]
                vector_mse = np.linalg.norm(self.label_dict[name])
                if vector_mse != 1.0:
                    print("{} is not unit vector, {}".format(name, vector_mse))
        #for name in self.label_id:
        #    diff = np.sum(np.square(np.subtract(self.label_dict[name], self.label_dict["jwpyo"])))
        #    print("Diff with {} and jwpyo: {}".format(name, diff))
        print("Finishing importing label file.")
        #print(self.label_dict)
    def crop_face(self, input_img):
        try:
            from edgetpu.detection.engine import DetectionEngine
        except:
            print("Cannot import detection library.")
            raise 
        def crop_with_bbox(objs, tensor):
            """
            crop the face image in input tensor.
            input:
                tensor: 1-D flattened camera input
                bbox: coordinate of bbox. range is [0,1]
                resize: (width, height) values that you want to resize the cropped face. Otherwise, assign it None
            returns:
                list of PIL image only contains cropped face.
            """
            #TODO
            ret = []
            tensor_3d = np.reshape(tensor, (320, 320, 3)) #coral camera module's input size is 320 * 320
            for obj in objs:
                x0, y0, x1, y1 = obj.bounding_box.flatten().tolist()
                x0, y0, x1, y1 = int(x0*320), int(y0*320), int(x1*320), int(y1*320)
                cropped = tensor_3d[x0:x1,y0:y1,:]
                img = Image.fromarray(np.uint8(cropped))    
                ret.append(img)

            return ret
        engine = DetectionEngine("/home/mendel/facenet/mobilenet_ssd_v2_face_quant_postprocess_edgetpu.tflite")
        img_pil = Image.open(input_img)
        origin_h, origin_w = img_pil.size[1], img_pil.size[0]
        print(origin_h, origin_w)
        img_tensor = np.array(img_pil).flatten()
        objs = engine.DetectWithInputTensor(img_tensor, threshold=10.0, top_k=3)
        cropped_faces = crop_with_bbox(objs, img_tensor)
        return NotImplemented
        return cropped_faces
    
    def prewhiten(self, x):
        mean = np.mean(x)
        std = np.std(x)
        std_adj = np.maximum(std, 1.0/np.sqrt(x.size))
        y = np.multiply(np.subtract(x, mean), 1/std_adj)
        return y
    
    def get_mean_std(self, input_tensor):
        num = reduce(lambda x, y: x * y, input_tensor.shape)
        mean = np.sum(input_tensor) / num
        std = np.sum(np.square(input_tensor - mean)) / num  
        return mean, std

    def normalize(self, input_tensor, opt="normal", range_ab=None):
        if opt == "normal":
            mean, std = self.get_mean_std(input_tensor)
            print("mean: {}, std: {}".format(mean, std))
        elif opt == "linear":
            """
            normalize variables to [a, b]
            """
            a, b = range_ab
            min_element, max_element = min(input_tensor), max(input_tensor)
            input_tensor = (b - a)*(input_tensor - min_element)/(max_element - min_element) + a
            return input_tensor
        return (input_tensor - mean) / std

    def GetEmbeddingVector(self, img):
        """
        Returns embedding vector from PIL image.
        inputs:
            img: PIL Image type(any * any * 3)
            (resized img may be 160* 160)
            normalize(stored_value): return unit vector if it is True
        returns:
            inf_time: inference time from img to embedding vector
            result: embedding vector(512, 1)
        """
        IMAGE_PREWHITEN = False
        VECTOR_NORMALIZE = True

        input_tensor_shape = self.get_input_tensor_shape()
        _, height, width, _ = input_tensor_shape
        output_tensor_shape = self.get_all_output_tensors_sizes()
        # resize
        img = img.resize((width, height), Image.BILINEAR)
        input_tensor = np.asarray(img).flatten()
        #print(input_tensor[0:10])
        if IMAGE_PREWHITEN:
            input_tensor = self.prewhiten(input_tensor)
            input_tensor = np.asarray([int((k/0.015686275)+64) for k in input_tensor_], dtype=np.uint8) 
        #print(input_tensor_[0:10])
        #import pdb; pdb.set_trace()
        #input_tensor = (self.normalize(input_tensor, opt="normal", range_ab=(0,255)) * 255).astype(np.uint8)
        #print("input tensor is {}".format(input_tensor))
        #dequantize from uint8[0, 255] to float32 [-1, 1]
        #input_tensor = (np.float32(input_tensor) - 127.5) / 128.0
        #print(input_tensor)
        #input_tensor = input_tensor/128.0 - 1
        #print("??")
        inf_time, result = self.RunInference(input_tensor)
        if VECTOR_NORMALIZE:
            return inf_time, img, result / np.linalg.norm(result)
        else:
            return inf_time, img, result

    def rankdata(self, a):
        n = len(a)
        ivec=sorted(range(len(a)), key=a.__getitem__)
        svec=[a[rank] for rank in ivec]
        sumranks = 0
        dupcount = 0
        newarray = [0]*n
        for i in range(n):
            sumranks += i
            dupcount += 1
            if i==n-1 or svec[i] != svec[i+1]:
                averank = sumranks / float(dupcount) + 1
                for j in range(i-dupcount+1,i+1):
                    newarray[ivec[j]] = averank
                sumranks = 0
                dupcount = 0
        return newarray 
    
    def CompareEV(self, ev, threshold=0.5, metric="L2", top_k = 1):
        """
        compare the corresponding embedding vector with anchor class' vector.
        returns:
            class_obj: name of person
        """
        inf_result = dict()
        L2_list = []
        L2_name = []
        #print("jwpyo size, ", np.linalg.norm(self.label_dict["jwpyo"]))
        #print("jhlee size, ", np.linalg.norm(self.label_dict["jhlee"]))
        if len(self.data_factory) > 0:
            for anchor_vec, name in self.data_factory:
                diff = np.subtract(anchor_vec, ev)
                L2 = np.linalg.norm(diff)
                L2_list.append(L2)
                L2_name.append(name)
            rank_list = [int(i) for i in self.rankdata(L2_list)]
            #print("L2_list: {}".format(L2_list))
            #print("rank_list: {}".format(rank_list))
        
        else: 
            for i in self.label_id.keys():
                name = self.label_id[i]
                if metric == "L2":
                    diff = np.subtract(self.label_dict[name], ev)
                    L2 = np.linalg.norm(diff) # sqrt(diff^2)
                    L2_list.append(L2)
                elif metric == "cosine":
                    diff = np.multiply(self.label_dict[name], ev)
                    L2 = diff
                    L2_list.append(L2)
                elif metric == "manhattan":
                    diff = np.subtract(self.label_dict[name], ev)
                    L2 = sum([abs(elem) for elem in diff])
                    L2_list.append(L2)
                #if L2 < threshold:
                    #print("{}, L2 is {}".format(name, L2))
                    #inf_name.append(name)
            rank_list = [int(i) for i in self.rankdata(L2_list)]
            print("L2_list: {}".format(L2_list))
            print("rank_list: {}".format(rank_list))
        
        if len(self.data_factory) > 0:
            try:
                for i in range(top_k):
                    top_k_results = [] # list for top-k distance and name
                    indices = Utils.list_index(rank_list, i+1)
                    for k in indices:
                        top_k_results.append((L2_name[k], float(round(L2_list[k], 4))))
                        if L2_name[k] in inf_result:
                            inf_result[L2_name[k]] += 1
                        else:
                            inf_result[L2_name[k]] = 1
                print(inf_result)
                ret = inf_result.copy()
                
                for key in inf_result:
                    if ret[key] < threshold:
                        del ret[key]
                return ret
            except ValueError:
                pass
        else:
            for i in range(top_k):
                indices = Utils.list_index(rank_list, i+1)
                for k in indices:
                    #inf_result[self.label_id[rank_list.index(i+1)]] = float(round(L2_list[rank_list.index(i+1)], 4))
                    inf_result[self.label_id[k]] = float(round(L2_list[k], 4))

        return inf_result
    def classify(self, classifier_path="/home/mendel/facenet/lab_classifier.pkl"):
        """
        print('Testing classifier')
        with open(classifier_path, 'rb') as infile:
            (model, class_names) = pickle.load(infile)

        print('Loaded classifier model from file "%s"' % classifier_path)

        predictions = model.predict_proba(emb_array)
        best_class_indices = np.argmax(predictions, axis=1)
        best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
        
        for i in range(len(best_class_indices)):
            print('%4d  %s: %.3f' % (i, class_names[best_class_indices[i]], best_class_probabilities[i]))
            
        accuracy = np.mean(np.equal(best_class_indices, labels))
        print('Accuracy: %.3f' % accuracy)
        """
        return NotImplemented
    
    def Camera(self, camera_module="gstream"):
        
        if camera_module == "gstream":
            
            def user_callback(img, svg_canvas):
                inf_time,_,ev = self.GetEmbeddingVector(img)
                text_lines = [
                    "Embedding vector: {}".format(ev),
                    "Inference time: {} ms".format(inf_time)
                ]
                Utils.generate_svg(svg_canvas, text_lines)
            result = gstreamer.run_pipeline(user_callback)
        """
        elif camera_module == "picamera":
            with picamera.PiCamera() as camera:
                camera.resolution = (640, 480)
                camera.framerate = 30
                _, width, height, channels = self.get_input_tensor_shape()
                camera.start_preview()
                try:
                    stream = io.BytesIO()
                    for foo in camera.capture_continuous(stream,
                                                         format='rgb',
                                                         use_video_port=True,
                                                         resize=(width, height)):
                        stream.truncate()
                        stream.seek(0)
                        input = np.frombuffer(stream.getvalue(), dtype=np.uint8)
                        results = self.GetEmbeddingVector(input)
                        if results:
                            camera.annotate_text = "Embedding vector: {}, elapsed time: {} ms".format(results[1], results[0])
                finally:
                    camera.stop_preview()
        """

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
      '--model', help='File path of Tflite model.', required=True)
    parser.add_argument(
      '--label-make', help="Make label file", required=False, type=bool)
    args = parser.parse_args()

    engine = FacenetEngine(args.model, None)
    """
    def user_callback(img, svg_canvas):
        results = engine.GetEmbeddingVector(img)
        text_lines = [
            "Embedding vector: {}".format(results[1][0]),
            "Inference time: {} ms".format(results[0])
        ]
        print(" ".join(text_lines))
        engine.generate_svg(svg_canvas, text_lines)
    result = gstreamer.run_pipeline(user_callback)
    """
    if args.label_make == True:
        engine.generate_labelfile(avg_only=False, per_class_img_num=1)
        import sys; sys.exit();
    #engine.classify()
    #engine.crop_face("/home/mendel/facenet/labmemberpic/jwpyo/IMG_20190208_085727.jpg")
    img = Image.open("/home/mendel/facenet/lfw_mtcnnalign_160/Zorica_Radovic/Zorica_Radovic_0001.png")
    _, _, ev = engine.GetEmbeddingVector(img)
    
    print("Embedding vector: ", ev)
    #print("Embedding vector value's statistic: ")
    #for elem in ev.tolist():
    #    print(str(elem), str(ev.tolist().count(elem)))
    import sys; sys.exit();
    engine.Camera()
    

if __name__ == "__main__":
    main()
