videos = r"D:\Python_Project\ultralytics-main\ultralytics\dataset\images\train\000040.jpg"
pt = r"D:\Python_Project\ultralytics-main\runs\detect\train5\weights\best.pt"
import cv2

confg = r"D:\Python_Project\ultralytics-main\runs\detect\train\args.yaml"
from ultralytics import YOLO
from PIL import Image

model = YOLO(pt)
# accepts all formats - image/dir/Path/URL/video/PIL/ndarray. 0 for webcam
# # results = model.predict(source=videos)
# results = model.predict(source=videos, show=True)  # Display preds. Accepts all YOLO predict arguments

# from PIL
# im1 = Image.open(r"D:\Python_Project\ultralytics-main\img.jpg")
# results = model.predict(source=im1, save=True)  # save plotted images
#
# # from ndarray
im2 = cv2.imread(r"C:\Users\周海君\Desktop\1111.png")
results = model.predict(source=im2, save=True, save_txt=True)  # save predictions as labels
#
# # from list of PIL/ndarray
# results = model.predict(source=[im1, im2])
# rgb_image = cv2.cvtColor(newImage, cv2.COLOR_BGR2RGB)
#         h, w, ch = rgb_image.shape
#         bytes_per_line = ch * w
#         qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
#         pixmap = QPixmap.fromImage(qt_image)
#         self.imgRightLabel.setPixmap(pixmap.scaled(self.imgLeftLabel.size(), QtCore.Qt.KeepAspectRatio))