# FOR DETAILED REFERENCE VISIT https://github.com/OlafenwaMoses/ImageAI


# ----------------------------------------------------------------------

# Image recognition model 
# from imageai.Classification import ImageClassification
# import os

# execution_path = os.getcwd()

# prediction = ImageClassification()
# prediction.setModelTypeAsDenseNet121()
# prediction.setModelPath(os.path.join(execution_path, "DenseNet-BC-121-32.h5"))
# prediction.loadModel()

# predictions, probabilities = prediction.classifyImage(os.path.join(execution_path, "2.png"), result_count=5 )
# for eachPrediction, eachProbability in zip(predictions, probabilities):
#     print(eachPrediction , " : " , eachProbability)



# ----------------------------------------------------------------------
# Image detection Model 
from imageai.Detection import ObjectDetection
import os

execution_path = os.getcwd()

detector = ObjectDetection()
detector.setModelTypeAsTinyYOLOv3()
detector.setModelPath( os.path.join(execution_path , "yolo-tiny.h5"))
detector.loadModel()
detections = detector.detectObjectsFromImage(input_image=os.path.join(execution_path , "4.png"), output_image_path=os.path.join(execution_path , "image4new.png"), minimum_percentage_probability=60)

for eachObject in detections:
    print(eachObject["name"] , " : ", eachObject["percentage_probability"], " : ", eachObject["box_points"] )
    print("--------------------------------")

