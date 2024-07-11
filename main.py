import os
import librosa
import numpy as np
import soundfile as sf
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from keras.models import Model
from keras.layers import GlobalAveragePooling1D, Attention, Input, Conv2D, MaxPooling2D, Flatten, Dense, LSTM, Bidirectional, TimeDistributed, Dropout, BatchNormalization
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
# Function to preprocess and segment the audio
def preprocess_and_segment(audio_path, output_dir, segment_duration=2, sr=16000):
    # Load audio file
    y, sr = librosa.load(audio_path, sr=sr, mono=True)
    
    # Normalize volume
    y = librosa.util.normalize(y)

    # Remove silence (optional)
    intervals = librosa.effects.split(y, top_db=20)
    y_trimmed = np.concatenate([y[start:end] for start, end in intervals])
    
    # Segment the trimmed audio into fixed-length chunks
    segment_samples = int(segment_duration * sr)  # Convert duration to samples
    num_segments = len(y_trimmed) // segment_samples  # Calculate the number of segments

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    segment_paths = []
    
    # Iterate over segments and save each one
    for i in range(num_segments):
        segment_start = i * segment_samples #Get the beginning of the segment
        segment_end = (i + 1) * segment_samples # Get the ending of the segment
        segment = y_trimmed[segment_start:segment_end] #Store the range segment
        #Concatenate file name
        segment_filename = f'{os.path.splitext(os.path.basename(audio_path))[0]}_segment_{i + 1}.wav' 
        #Find output path
        output_path = os.path.join(output_dir, segment_filename)
        #Store the segment
        sf.write(output_path, segment, sr)
        segment_paths.append(output_path)
    
    # Handle the last segment if it is shorter than segment_duration
    if len(y_trimmed) % segment_samples != 0:
        segment = y_trimmed[num_segments * segment_samples:] #store whatever left 
        segment_filename = f'{os.path.splitext(os.path.basename(audio_path))[0]}_segment_{num_segments + 1}.wav'
        output_path = os.path.join(output_dir, segment_filename)
        sf.write(output_path, segment, sr)
        segment_paths.append(output_path)
    
    return segment_paths

# Base directory for the videos
base_dir = 'training/training/drive-download-20240522T165804Z-001'
test_base_dir = 'testing/testing\drive-download-20240522T165559Z-001'

# List to store file paths and their corresponding labels
data = []
testData=[]

# Loop through each directory (accent)
for accent in os.listdir(base_dir):
    accent_dir = os.path.join(base_dir, accent)
    if os.path.isdir(accent_dir):
        # Loop through each audio file in the accent directory
        for file in os.listdir(accent_dir): 
            if file.endswith('.wav') and not file.startswith('segment_'): #check if its already been segmented

                audio_path = os.path.join(accent_dir, file) #create the path to the file
                output_dir = os.path.join(accent_dir, 'segments') #store the file in the segments directory
                segment_paths = preprocess_and_segment(audio_path, output_dir) #segment audio file
                
                # Append the segment paths and labels to the data list
                for segment_path in segment_paths:
                    data.append({'path': segment_path, 'label': accent})

for accent in os.listdir(test_base_dir):
    accent_dir = os.path.join(test_base_dir, accent)
    if os.path.isdir(accent_dir):
        # Loop through each audio file in the accent directory
        for file in os.listdir(accent_dir):
            if file.endswith('.wav') and not file.startswith('segment_'):
                audio_path = os.path.join(accent_dir, file)
                output_dir = os.path.join(accent_dir, 'segments')
                segment_paths = preprocess_and_segment(audio_path, output_dir)
                
                # Append the segment paths and labels to the data list
                for segment_path in segment_paths:
                    testData.append({'path': segment_path, 'label': accent})

def extract_mfcc(audio_path, sr=16000, n_mfcc=13, n_fft=1024, hop_length=256, n_mels=128, fmin=0, fmax=None):
    y, sr = librosa.load(audio_path, sr=sr, mono=True)
    
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, fmin=fmin, fmax=fmax)
    
    mfcc = mfcc.T
    
    return mfcc

# List to store MFCC features and their corresponding labels
mfcc_features = []
labels = []
test_mfcc_features=[]
test_labels=[]
# Extract MFCC features for each segment
for item in data:
    mfcc = extract_mfcc(item['path'])
    mfcc_features.append(mfcc)
    labels.append(item['label'])

for item in testData:
    mfcc = extract_mfcc(item['path'])
    test_mfcc_features.append(mfcc)
    test_labels.append(item['label'])

# Convert to numpy arrays for easier handling
mfcc_features = np.array(mfcc_features, dtype=object)  # Using dtype=object to handle different sequence lengths
labels = np.array(labels)
test_mfcc_features = np.array(test_mfcc_features, dtype=object)
test_labels = np.array(test_labels)
# Print shapes for verification
print(f"MFCC features shape: {mfcc_features.shape}")
print(f"Labels shape: {labels.shape}")

def build_hybrid_cnn_las_model(input_shape, num_classes):
    input_layer = Input(shape=input_shape)
    # CNN layers
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_layer)
    x = MaxPooling2D((2, 2))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)    

    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)  # Reduced number of filters
    x = MaxPooling2D((2, 2))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)    

    x = TimeDistributed(Flatten())(x)

    # LAS layers
    x = Bidirectional(LSTM(128, return_sequences=True))(x)  # Reduced number of units
    x = Dropout(0.3)(x)
    y = Bidirectional(LSTM(128))(x)
    y = Dropout(0.3)(y)

    attention_out = Attention()([x,y])
    pooled_out = GlobalAveragePooling1D()(attention_out)

    output_layer = Dense(num_classes, activation='softmax')(pooled_out)
    
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model

max_len = max(len(mfcc) for mfcc in mfcc_features)
max_len_test = max(len(mfcc) for mfcc in test_mfcc_features)
mfcc_features_padded = pad_sequences(mfcc_features, maxlen=max_len, padding='post', dtype='float32')
test_mfcc_features_padded = pad_sequences(test_mfcc_features, maxlen=max_len, padding='post', dtype='float32')

label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)
labels_one_hot = to_categorical(labels_encoded)
test_labels_encoded = label_encoder.fit_transform(test_labels)
test_labels_one_hot = to_categorical(test_labels_encoded)

X_train = mfcc_features_padded
y_train = labels_one_hot

X_val = test_mfcc_features_padded
y_val = test_labels_one_hot
X_train_reshaped = np.expand_dims(X_train, -1)
X_val_reshaped = np.expand_dims(X_val, -1)

input_shape = (max_len, 13, 1)  # (time, n_mfcc, 1) for 2D convolutions
num_classes = len(label_encoder.classes_)

# Use a lower learning rate for the Adam optimizer
optimizer = Adam(learning_rate=0.0001)
model = build_hybrid_cnn_las_model(input_shape, num_classes)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Print model summary
model.summary()

# Train the model with increased batch size
history = model.fit(X_train_reshaped, y_train, validation_data=(X_train_reshaped, y_train), epochs=30, batch_size=32)

test_loss, test_accuracy = model.evaluate(X_val_reshaped, y_val)
print(f"Test loss: {test_loss}, Test accuracy: {test_accuracy}")

# Make predictions
y_pred = model.predict(X_val_reshaped)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_val, axis=1)

# Calculate precision, recall, and F1-score
precision = precision_score(y_true_classes, y_pred_classes, average='weighted')
recall = recall_score(y_true_classes, y_pred_classes, average='weighted')
f1 = f1_score(y_true_classes, y_pred_classes, average='weighted')

print(f"Test Precision: {precision}")
print(f"Test Recall: {recall}")
print(f"Test F1-Score: {f1}")

# Generate a classification report
report = classification_report(y_true_classes, y_pred_classes, target_names=label_encoder.classes_)
print(report)