import ultralytics

from ultralytics import YOLO

model = YOLO("model\gall_detectv5.pt")

result = model.predict(source='test_images\IMG_20231121_162619.jpg', conf=0.25, save=True)