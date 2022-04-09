# ===========================================
# Face Detection using Cascade Classifier
# ===========================================
import cv2
import os
from os import listdir

# load classifier
upper_body = cv2.CascadeClassifier(
    "D:/SKRIPSI/haarcascadesXML/haarcascade_upperbody.xml")
folder_dir = "D:/SKRIPSI/Kagglehumandataset/filtered/"
# folder_dir = "D:/SKRIPSI/PNGImages/"
for images in os.listdir(folder_dir):

    # check data dengan akhiran PNG
    if (images.endswith(".png")):

        # get image to variable
        path = "D:/SKRIPSI/Kagglehumandataset/filtered/"
        imageName = images
        img = cv2.imread(path+imageName)
        print(img)

        # pre-processing
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        upper = upper_body.detectMultiScale(
            gray_img,
            scaleFactor=1.1,
            minNeighbors=3
        )

        # ===========================================
        # menambahkan kotak deteksi di area person
        # ===========================================

        for x, y, w, h in upper:
            img = cv2.rectangle(
                img,            # image object
                (x, y),          # posisi kotak
                (x+w, y+h),     # posisi kotak
                (128, 255, 0),    # warna kotak RGB
                3               # lebar garis kotak
            )

        resized = cv2.resize(img, (800, 600))
        # cv2.imshow('Gambar Output', resized)
        cv2.imwrite('D:/SKRIPSI/fromImageHaar/outputfiltered20/' +
                    images, img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
