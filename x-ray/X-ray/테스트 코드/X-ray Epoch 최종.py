import os
import glob
import sys
assert sys.version_info >= (3, 5) # Python 버전이 3.5 이상인지 확인 (아니면 AssertionError 발생)
import tensorflow as tf
from keras.applications import ResNet50V2
import numpy as np
import matplotlib.pyplot as plt
import koreanize_matplotlib
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import keras
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Sequential
import sklearn
assert sklearn.__version__ >= "0.20" # Scikit-learn 버전이 0.20 이상인지 확인
from sklearn.metrics import classification_report

main_path = "C:/Users/myong/Desktop/x-ray/input/archive/chest_xray/chest_xray" # 저장 위치

train_path = os.path.join(main_path, "train") # 훈련용 이미지
test_path = os.path.join(main_path,  "test") # 테스트용 이미지
val_path = os.path.join(main_path, "val") # 검증용 이미지 
 
pneumonia_train_images = glob.glob(train_path+"/PNEUMONIA/*.jpeg") # 훈련용 폐렴 이미지
normal_train_images = glob.glob(train_path+"/NORMAL/*.jpeg") # 훈련용 정상 이미지

pneumonia_val_images = glob.glob(val_path+"/PNEUMONIA/*.jpeg") # 검증용 폐렴 이미지
normal_val_images = glob.glob(val_path+"/NORMAL/*.jpeg") # 검증용 정상 이미지

pneumonia_test_images = glob.glob(test_path+"/PNEUMONIA/*.jpeg") # 테스트용 폐렴 이미지
normal_test_images = glob.glob(test_path+"/NORMAL/*.jpeg") # 테스트용 폐렴 이미지

# 검증용 데이터 개수 출력
val_Pneumonia = len(os.listdir(val_path+'/PNEUMONIA'))
val_Normal = len(os.listdir(val_path+'/NORMAL'))
print(f'len(val_Normal) = {val_Normal}, len(val_Pneumonia)={val_Pneumonia}')

# 디렉토리 복사
from setuptools._distutils.dir_util import copy_tree

copy_tree(main_path, 'temp')
path = "C:/Users/myong/Desktop/x-ray/input/archive/chest_xray/chest_xray"

train_path = os.path.join(path,"train")
test_path = os.path.join(path,"test")
val_path = os.path.join(path,"val")

val_Pneumonia = len(os.listdir(val_path+'/PNEUMONIA'))
val_Normal = len(os.listdir(val_path+'/NORMAL'))
print(f'len(val_Normal) = {val_Normal},len(val_Pneumonia)={val_Pneumonia}')

train_datagen = ImageDataGenerator(
    rescale=1/255
)
val_datagen = ImageDataGenerator(
    rescale=1/255
)


input_shape = (512, 512, 3)

base_model = tf.keras.applications.ResNet50V2(weights = 'imagenet', input_shape = input_shape, include_top=False)

for layer in base_model.layers:
    layer.trainable = False
    
model = Sequential()
model.add(base_model)
model.add(GlobalAveragePooling2D())
model.add(Dense(128, activation='relu', kernel_regularizer=keras.regularizers.l2(0.1)))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid')) # 시그모이드 함수 
model.summary()

train_Datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)

# 훈련용 데이터 생성기
train_generator = train_Datagen.flow_from_directory(
    train_path,
    target_size = (512,512),
    batch_size= 64,
    class_mode = 'binary'
)

# 검증용 데이터 생성기
validation_generator = val_datagen.flow_from_directory(
    test_path,
    target_size = (512,512),
    batch_size= 64,
    class_mode = 'binary'
)

# 테스트용 데이터 생성기
test_generator = val_datagen.flow_from_directory(
    val_path,
    target_size = (512,512),
    batch_size= 64,
    class_mode = 'binary'
)

from keras.optimizers import Adam
from keras.callbacks import EarlyStopping

initial_learning_rate = 1e-3
Ir_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate,
    decay_steps=755,
    decay_rate=0.9,
    staircase=True
)

model.compile(optimizer=Adam(Ir_schedule), loss='binary_crossentropy', metrics=["accuracy"])

steps_per_epoch = len(train_generator) // train_generator.batch_size
validation_steps = len(validation_generator) // validation_generator.batch_size

# 모델 학습 수행
history = model.fit(train_generator,
                    epochs=10,
                    steps_per_epoch= 5216 // 64,
                    validation_data= validation_generator,
                    validation_steps= 624 // 64)

model.evaluate(test_generator)[1] # 테스트정확도 확인하기

model.evaluate(validation_generator)[1] # 검증정확도 확인하기

from keras.callbacks import EarlyStopping
es = EarlyStopping(monitor = 'val_loss', mode = 'auto', verbose = 1)

history_new = model.fit(train_generator,
                        epochs=10
                        , # ex) epochs가 5면 early stopping이라고 출력하면서 멈춤
                        steps_per_epoch= 5216 // 64,
                        validation_data= validation_generator,
                        validation_steps= 624 // 64,
                        callbacks=[es])

accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(15,10))
plt.subplot(2, 2, 1)
plt.plot(accuracy, label = "Training accuracy")
plt.plot(val_accuracy, label="Validation accuracy")
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.ylim(0.5, 1)
plt.legend()
plt.title("Training vs validation accuracy")

plt.subplot(2, 2, 2)
plt.plot(loss, label = "Training loss")
plt.plot(val_loss, label="Validation loss")
plt.ylabel('loss')
plt.xlabel('epoch')
plt.ylim(0, 1)
plt.legend()
plt.title("Training vs validation loss")

plt.show() 

model.evaluate(validation_generator)[1] # 검증용 정확도 확인
model.evaluate(test_generator)[1] # 테스트용 정확도 확인

y_true = test_generator.classes  # 실제 레이블 가져오기
y_pred = model.predict(test_generator)  # 모델 예측
y_pred = np.round(y_pred).astype(int).reshape(-1)  # 예측된 값을 0과 1로 변환
print(classification_report(y_true, y_pred))