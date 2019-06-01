from edgetpu.classification.engine import ClassificationEngine
from edgetpu.utils.image_processing import ResamplingWithOriginalRatio 
import numpy as np
from PIL import Image
import argparse
import math
from functools import reduce
#import picamera, io
import gstreamer, svgwrite, imp
import time

class FacenetEngine(ClassificationEngine):

    def __init__(self, model_path, device_path=None):
        if device_path:
            super().__init__(model_path, device_path)
        else:
            super().__init__(model_path)
    def generate_labelfile(self, data_folder="/home/mendel/facenet/labmemberpic/", save_as="/home/mendel/coral_utils/labels.txt"):
        """
        generate embedding vector label file with corresponding directory.
        """
        import glob
        members = glob.glob(data_folder+"*")
        with open(save_as, "w") as write_file:
            for name_folder in members:
                images = glob.glob(name_folder+"/*")
                avg_ev = np.zeros(512, )
                for img_path in images:
                    img = Image.open(img_path)
                    _, ev = self.GetEmbeddingVector(img)
                    avg_ev = avg_ev + ev
                    break
                #avg_ev = avg_ev / len(images)
                print(avg_ev)
                write_file.write(name_folder.split("/")[-1] + " ")
                for i, elem in enumerate(avg_ev):
                    if i == len(avg_ev) - 1:
                        write_file.write(str(elem))
                    else:
                        write_file.write(str(elem)+ ",")
                write_file.write("\n")

    def ImportLabel(self, label_file="/home/mendel/coral_utils/models/labels.txt"):
        """
        jwpyo, [1, 2, ... ]
        """
        self.label_dict = dict()
        self.label_id = list()
        with open(label_file, "r") as lf:
            for line in lf:
                class_name, raw_ev = line.split(" ")
                ev = list(map(float, raw_ev.split(",")))
                self.label_dict[class_name] = ev
                self.label_id.append(class_name)
        #check if it is unit vector
        for name in self.label_id:
            vector_sum = np.sum(np.square(self.label_dict[name]))
            if vector_sum != 1.0:
                print("{} is not unit vector, {}".format(name, vector_sum))
        for name in self.label_id:
            diff = np.sum(np.square(np.subtract(self.label_dict[name], self.label_dict["jwpyo"])))
            print("Diff with {} and jwpyo: {}".format(name, diff))
        print("Finishing importing label file.")
        #print(self.label_dict)
    def get_mean_std(self, input_tensor):
        num = reduce(lambda x, y: x * y, input_tensor.shape)
        mean = np.sum(input_tensor) / num
        std = np.sum(np.square(input_tensor - mean)) / num  
        return mean, std

    def normalize(self, input_tensor, opt="normal", range_ab=None):
        if opt == "normal":
            mean, std = self.get_mean_std(input_tensor)
        elif opt == "linear":
            """
            normalize variables to [a, b]
            """
            a, b = range_ab
            min_element, max_element = min(input_tensor), max(input_tensor)
            input_tensor = (b - a)*(input_tensor - min_element)/(max_element - min_element) + a
            return input_tensor
        return (input_tensor - mean) / std + 0.5

    def GetEmbeddingVector(self, img):
        """
        Returns embedding vector from PIL image.
        inputs:
            img: PIL Image type(180 * 180 * 3)

        returns:
            inf_time: inference time from img to embedding vector
            result: embedding vector(512, 1)
        """
        input_tensor_shape = self.get_input_tensor_shape()
        _, height, width, _ = input_tensor_shape
        img = img.resize((width, height), Image.NEAREST)
        input_tensor = np.asarray(img).flatten()
        #input_tensor = (self.normalize(input_tensor, opt="normal", range_ab=(0,255)) * 255).astype(np.uint8)
        #print("input tensor is {}".format(input_tensor))
        #dequantize from uint8[0, 255] to float32 [-1, 1]
        #input_tensor = (np.float32(input_tensor) - 127.5) / 128.0
        #print(input_tensor)
        #input_tensor = input_tensor/128.0 - 1
        inf_time, result = self.RunInference(input_tensor)
        return inf_time, self.normalize(result, opt="linear", range_ab=(-1,1))

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
    
    def CompareEV(self, ev, threshold=3.99, top_k = 3):
        """
        compare the corresponding embedding vector with anchor class' vector.
        returns:
            class_obj: name of person
        """
        inf_name = []
        L2_list = []
        for name in self.label_id:
            diff = np.subtract(self.label_dict[name], ev)
            #print(diff)
            L2 = np.linalg.norm(diff) # sqrt(diff^2)
            L2_list.append(L2)
            if L2 < threshold:
                print("{}, L2 is {}".format(name, L2))
                #inf_name.append(name)
        rank_list = [int(i) for i in self.rankdata(L2_list)]
        for i in range(top_k):
            inf_name.append(self.label_id[rank_list.index(i+1)])
        return inf_name

        
        return NotImplemented
    @staticmethod
    def L2distance(v1, v2):
        assert(len(v1) == len(v2))
        dis = 0
        for e1, e2 in zip(v1, v2):
            e = (e1 - e2) ** 2
            dis += e
        return sqrt(dis)

    @staticmethod
    def generate_svg(dwg, text_lines):
        for y, line in enumerate(text_lines):
            dwg.add(dwg.text(line, insert=(11, (y+1)*20+1), fill='white', font_size='20'))
            dwg.add(dwg.text(line, insert=(11, (y+1)*20), fill='white', font_size='20'))
    
    def Camera(self, camera_module="gstream"):
        
        if camera_module == "gstream":
            
            def user_callback(img, svg_canvas):
                results = self.GetEmbeddingVector(img)
                text_lines = [
                    "Embedding vector: {}".format(results[1]),
                    "Inference time: {} ms".format(results[0])
                ]
                self.generate_svg(svg_canvas, text_lines)
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
    #engine.generate_labelfile()
    engine.Camera()
    

if __name__ == "__main__":
    main()
