from transformers import PrtrFeatureExtractor, PrtrForObjectDetection
from PIL import Image
import requests
from transformers import PrtrConfig
import matplotlib.pyplot as plt


feature_extractor = PrtrFeatureExtractor.from_pretrained("anugunj/prtr-resnet50-384x288")
model = PrtrForObjectDetection.from_pretrained("anugunj/prtr-resnet50-384x288")

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

inputs = feature_extractor(images=image, return_tensors="pt")

outputs = model(**inputs)