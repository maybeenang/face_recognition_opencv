#  buat fungsi untuk mengambil dataset dari kamera

import cv2
import numpy as np
from PIL import Image
import os

# path untuk dataset
path = 'dataset'

# recognizer untuk LBPH
recognizer = cv2.face.LBPHFaceRecognizer_create()

# detector wajah menggunakan haarcascade
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml");

# inputan untuk nama dan buat id auto increment
nama = input("Masukkan nama: ")

# ambil id terakhir
last_id = 0
for file in os.listdir(path):
    file_id = int(file.split(".")[1])
    if file_id > last_id:
        last_id = file_id

# id baru
id_baru = last_id + 1

# buat folder dataset jika belum ada
if not os.path.exists(path):
    os.makedirs(path)

print("\n [INFO] Sedang menganmbil dataset wajah, tunggu sebentar ...")
# ambil gambar dari kamera jika ada wajah terdeteksi sebanyak 30 kali
cam = cv2.VideoCapture(0)
cam.set(3, 640) # set video width
cam.set(4, 480) # set video height

count = 0

while(True):
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray,scaleFactor=1.3, minNeighbors=5)

    # looping melalui semua wajah dan tambahkan ke dalam list wajah
    for (x,y,w,h) in faces:
        count += 1

        # simpan gambar wajah yang terdeteksi
        cv2.imwrite("dataset/User." + str(id_baru) + '.' + str(count) + ".jpg", gray[y:y+h,x:x+w])

    # jika menekan tombol ESC maka keluar dari program
    k = cv2.waitKey(100) & 0xff 
    if k == 27:
        break
    # jika sudah mengambil 30 gambar maka keluar dari program
    elif count >= 30:
        break

# tampilkan jumlah gambar yang sudah diambil
print("\n [INFO] " + str(count) + " gambar telah diambil")

# masukan nama dan id ke dalam file database.csv
with open('database.csv', 'a') as f:
    f.write("%s,%s\n"%(nama,id_baru))

# tutup kamera
cam.release()

# tutup semua window
cv2.destroyAllWindows()

# fungsi untuk mengambil gambar dan label
def getImagesAndLabels(path):
        
    # mengambil semua file dalam folder dataset
    imagePaths = [os.path.join(path,f) for f in os.listdir(path)] 
    
    # inisialisasi wajah dan id
    faceSamples=[]
    ids = []

    # looping melalui semua gambar dan load gambar
    for imagePath in imagePaths:

        # konversi gambar ke dalam format grayscale
        PIL_img = Image.open(imagePath).convert('L')

        # konversi gambar ke dalam format numpy array
        img_numpy = np.array(PIL_img,'uint8')

        # mengambil simpan data wajah menurut nama file
        id = int(os.path.split(imagePath)[-1].split(".")[1])

        # mengambil wajah dari gambar
        faces = detector.detectMultiScale(img_numpy)

        # looping melalui semua wajah dan tambahkan ke dalam list wajah
        for (x,y,w,h) in faces:
            faceSamples.append(img_numpy[y:y+h,x:x+w])
            ids.append(id)

    # kembalikan wajah dan id
    return faceSamples,ids

# ambil wajah dan id
faces,ids = getImagesAndLabels(path)

# train recognizer
recognizer.train(faces, np.array(ids))

# simpan model ke dalam file trainer.yml
recognizer.write('trainer.yml')

# tampilkan jumlah wajah yang sudah di train
print("\n [INFO] " + str(len(np.unique(ids))) + " wajah telah di train. Keluar dari program")

