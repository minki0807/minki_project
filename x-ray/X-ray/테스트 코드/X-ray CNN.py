import os
import sys
assert sys.version_info >= (3, 5)  # Python 버전이 3.5 이상인지 확인
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import koreanize_matplotlib
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report
import sklearn
assert sklearn.__version__ >= "0.20"  # Scikit-learn 버전 확인

# 경로 설정
main_path = "C:/Users/myong/Desktop/x-ray/input/archive/chest_xray/chest_xray"
train_path = os.path.join(main_path, "train")
test_path = os.path.join(main_path, "test")
val_path = os.path.join(main_path, "val")

# 데이터 개수 출력
val_Pneumonia = len(os.listdir(val_path + '/PNEUMONIA'))
val_Normal = len(os.listdir(val_path + '/NORMAL'))
print(f'len(val_Normal) = {val_Normal}, len(val_Pneumonia) = {val_Pneumonia}')

# 데이터 생성기 설정
datagen_args = dict(rescale=1/255)
train_datagen = ImageDataGenerator(**datagen_args, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
val_datagen = ImageDataGenerator(**datagen_args)

# 데이터 로드
train_generator = train_datagen.flow_from_directory(
    train_path, target_size=(128, 128), batch_size=128, class_mode='binary')

validation_generator = val_datagen.flow_from_directory(
    val_path, target_size=(128, 128), batch_size=128, class_mode='binary')

test_generator = val_datagen.flow_from_directory(
    test_path, target_size=(128, 128), batch_size=128, class_mode='binary')

# CNN 모델 구성
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')  # 이진 분류이므로 sigmoid 사용
])

# 모델 컴파일
model.compile(optimizer=Adam(learning_rate=tf.keras.optimizers.schedules.ExponentialDecay(
                    initial_learning_rate=1e-3, decay_steps=755, decay_rate=0.9, staircase=True)),
              loss='binary_crossentropy', metrics=["accuracy"])

# 학습 및 EarlyStopping 설정
es = EarlyStopping(monitor='val_loss', patience = 10, verbose=1, mode='auto', restore_best_weights=True) 

history = model.fit(train_generator,
                    epochs=50,
                    validation_data=validation_generator,
                    callbacks=[es])

# 평가 및 정확도 출력
print("테스트 데이터 정확도:", model.evaluate(test_generator)[1])
print("검증 데이터 정확도:", model.evaluate(validation_generator)[1])

# 학습 결과 시각화
accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(15, 5))

plt.subplot(1, 2, 1)
plt.plot(accuracy, label="Training accuracy")
plt.plot(val_accuracy, label="Validation accuracy")
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.ylim(0.5, 1)
plt.legend()
plt.title("Training vs Validation Accuracy")

plt.subplot(1, 2, 2)
plt.plot(loss, label="Training Loss")
plt.plot(val_loss, label="Validation Loss")
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.ylim(0, 1)
plt.legend()
plt.title("Training vs Validation Loss")

plt.tight_layout()
plt.show()

# 예측 및 분류 리포트 출력
y_true = test_generator.classes
y_pred = (model.predict(test_generator) > 0.5).astype("int32").reshape(-1)
print(classification_report(y_true, y_pred))
