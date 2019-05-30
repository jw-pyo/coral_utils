from edgetpu.classification.engine import ClassificationEngine
from edgetpu.utils.image_processing import ResamplingWithOriginalRatio 
import numpy as np
from PIL import Image
import argparse

#import picamera, io
import gstreamer, svgwrite, imp
import time

class FacenetEngine(ClassificationEngine):

    def __init__(self, model_path, device_path=None):
        if device_path:
            super().__init__(model_path, device_path)
        else:
            super().__init__(model_path)

    def GetEmbeddingVector(self, img):
        """
        Returns embedding vector from PIL image.
        inputs:
            img: PIL Image type(180 * 180 * 3)

        returns:
            inf_time: inference time from img to embedding vector
            result: embedding vector(512, 1)
        """
        #TODO: resize the image size to 180*180
        input_tensor_shape = self.get_input_tensor_shape()
        _, height, width, _ = input_tensor_shape
        img = img.resize((width, height), Image.NEAREST)
        input_tensor = np.asarray(img).flatten()
        
        inf_time, result = self.RunInference(input_tensor)

        return inf_time, result
    
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
    engine.Camera()
    

if __name__ == "__main__":
    main()
