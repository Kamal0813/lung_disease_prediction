import pandas as pd
import numpy as np
from keras.layers import Conv2D, Dense, MaxPooling2D, Flatten, BatchNormalization, Dropout
from keras.models import Sequential
from keras.applications import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.applications.resnet import preprocess_input
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report,precision_score,recall_score,f1_score,confusion_matrix


# Dataset 1
df = pd.read_csv('/content/lung_cancer.csv')

encorder = LabelEncoder()
dataframe_col = df[["GENDER","SMOKING", "YELLOW_FINGERS", "ANXIETY", "PEER_PRESSURE", 'CHRONIC_DISEASE', "FATIGUE", "ALLERGY", 'WHEEZING',
                   'ALCOHOL_CONSUMING', 'COUGHING', 'SHORTNESS_OF_BREATH',
                    'SWALLOWING_DIFFICULTY', 'CHEST_PAIN', 'LUNG_CANCER']]

for col in dataframe_col.columns:
  df[col] = encorder.fit_transform(df[col])

for col in dataframe_col.columns:
  print(df[col].value_counts())

#Plotting lung cancer ratio
plt.pie(
    df['LUNG_CANCER'].value_counts(),
    labels = df['LUNG_CANCER'].value_counts().index,
    autopct = '%1.1f%%'
)

plt.title("Lung_Cancer")
plt.show()
#Plotting smoking ratio
plt.pie(
    df['SMOKING'].value_counts(),
    labels = df['SMOKING'].value_counts().index,
    autopct = '%1.1f%%'
)
plt.title("Smoking")
plt.show()

df.info()

descip = df[['AGE', "SMOKING", "YELLOW_FINGERS", "ANXIETY", "PEER_PRESSURE", 'CHRONIC_DISEASE', "FATIGUE", "ALLERGY", 'WHEEZING',
             'ALCOHOL_CONSUMING', 'COUGHING', 'SHORTNESS_OF_BREATH',
             'SWALLOWING_DIFFICULTY', 'CHEST_PAIN', 'LUNG_CANCER']]
descip.describe()

fig, axes = plt.subplots(nrows=7, ncols=2, figsize=(16, 32))
axes = axes.flatten()
for idx, col in enumerate(descip.columns):
  if idx < len(axes):
    sns.histplot(x=col, data=descip, ax=axes[idx], hue='LUNG_CANCER', bins=20, multiple='stack')
plt.tight_layout()
plt.show()

# plotting graph of lung cancer and age
lung_cancer=df[df['LUNG_CANCER']==1]
plt.figure(figsize=(10,5))
sns.histplot(lung_cancer,x=df['AGE'],hue='LUNG_CANCER',bins=30)
plt.title('Lung Cancer')
plt.show()

# Heatmap of the data
numeric_col=df.select_dtypes(include=[np.number])
corr=numeric_col.corr()
plt.figure(figsize=(12,10))
sns.heatmap(corr,annot=True,linewidths=0.5,fmt=".3f",cmap='viridis')
plt.show()

# preparing data for training and testing
x = df.drop(columns='LUNG_CANCER')
y = df['LUNG_CANCER']
x.head()

y.head()

# data splitting
x, y = make_classification(n_samples=500, n_features=15, random_state=42)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
scalar = StandardScaler()
x_train = scalar.fit_transform(x_train)
x_test = scalar.transform(x_test)
# using xgbclassifier
xgbclassifier=XGBClassifier(use_label_encoder=False, eval_metric='logloss',random_state=42)
xgbclassifier.fit(x_train,y_train)
y_pred=xgbclassifier.predict(x_test)

# calculation of accuracy, precision, recall, f1-score
accuracy = accuracy_score(y_test, y_pred) * 100
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}% | Precision: {precision:.4f} | Recall: {recall:.4f} | F1-Score: {f1:.4f}")

# confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(4, 3))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Negative", "Positive"], yticklabels=["Negative", "Positive"])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix - XGBClassifier")
plt.show()

import pickle
with open('lung_cancer_xgb_model.pkl', 'wb') as f:
    pickle.dump(xgbclassifier, f)

print("numpy", np.__version__)
print("pandas", pd.__version__)
print("seaborn", sns.__version__)

# Trying different models
models={
"LogisticRegression" : LogisticRegression(),
"RandomForest" : RandomForestClassifier(n_estimators=100, random_state=42),
"XGBClassifier" : XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
"KNN":KNeighborsClassifier(n_neighbors=10),
"DecisionTree":DecisionTreeClassifier(random_state=42),
"GradientBoosting": GradientBoostingClassifier(random_state=42),
" AdaBoostClassifier":AdaBoostClassifier(random_state=42)
}
for name,model in models.items():
  model.fit(x_train, y_train)
  y_pred = model.predict(x_test)

  accuracy = accuracy_score(y_test, y_pred) * 100
  precision = precision_score(y_test, y_pred)
  recall = recall_score(y_test, y_pred)
  f1 = f1_score(y_test, y_pred)
  print(f"Model: {name} | Accuracy: {accuracy:.2f}% | Precision: {precision:.4f} | Recall: {recall:.4f} | F1-Score: {f1:.4f}")

gb_classifier = GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=3,
    random_state=42
)
gb_classifier.fit(x_train, y_train)

y_pred = gb_classifier.predict(x_test)

# calculation of accuracy, precision, recall, f1-score
accuracy = accuracy_score(y_test, y_pred) * 100
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}% | Precision: {precision:.4f} | Recall: {recall:.4f} | F1-Score: {f1:.4f}")

with open('lung_cancer_gb_model.pkl', 'wb') as f:
    pickle.dump(gb_classifier, f)

# Dataset 2
train_path_str = r'D:\Sem1\Project\Lung_detection\model\Chest_CTScan_images\data\train'
val_path_str = r'D:\Sem1\Project\Lung_detection\model\Chest_CTScan_images\data\valid'
test_path_str = r'D:\Sem1\Project\Lung_detection\model\Chest_CTScan_images\data\test'

input_shape = (224, 224, 3)
num_classes = 4

trainGenertor = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=10,
    width_shift_range=0.3,
    height_shift_range=0.3,
    shear_range=0.2,
    zoom_range=0.1,
    horizontal_flip=True,
    vertical_flip=True,
)

valGenertor = ImageDataGenerator(preprocessing_function=preprocess_input)
testGenertor = ImageDataGenerator(preprocessing_function=preprocess_input)

train_data = trainGenertor.flow_from_directory(
    train_path_str, target_size=(224, 224), batch_size=16, class_mode='categorical'
)
val_data = valGenertor.flow_from_directory(
    val_path_str, target_size=(224, 224), batch_size=16, class_mode='categorical'
)
test_data = testGenertor.flow_from_directory(
    test_path_str, target_size=(224, 224), batch_size=16,
    class_mode='categorical', shuffle=False
)


VGG16_model = VGG16(include_top=False, weights='imagenet', input_shape=input_shape)
VGG16_model.trainable = False

model = Sequential([
    VGG16_model,
    Flatten(),
    BatchNormalization(),
    Dense(1024, activation='relu'),
    Dropout(0.3),
    Dense(512, activation='relu'),
    Dropout(0.3),
    Dense(256, activation='relu'),
    Dense(128, activation='relu'),
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

callbacks = [
    EarlyStopping(patience=7, restore_best_weights=True),
    ModelCheckpoint("best_model.keras", save_best_only=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3)
]

results = model.fit(
    train_data,
    validation_data=val_data,
    epochs=50,
    callbacks=callbacks,
    verbose=1
)


model.save("model.keras")


with open("model.keras", "rb") as file:
    print("Model file saved successfully:", file.name)









