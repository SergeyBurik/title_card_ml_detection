import os
import cv2
import json
import torch
import librosa
import numpy as np
import pandas as pd
import torchvision.models as tv_models
import torchvision.transforms as tv_transforms
from torchvggish import vggish, vggish_input
from sklearn.cluster import DBSCAN
from moviepy.editor import VideoFileClip
from tqdm import tqdm
from dtw import accelerated_dtw
from collections import defaultdict
from glob import glob

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load ResNet
resnet_model = tv_models.resnet18(pretrained=True)
resnet_model.fc = torch.nn.Identity()
resnet_model.eval().to(device)

image_transform = tv_transforms.Compose([
    tv_transforms.ToPILImage(),
    tv_transforms.Resize((224, 224)),
    tv_transforms.ToTensor(),
    tv_transforms.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
])

# Load VGGish
audio_model = vggish()
audio_model.eval().to(device)

def get_visual_features(video_file_path, frame_rate=1.0):
    video_capture = cv2.VideoCapture(video_file_path)
    video_fps = video_capture.get(cv2.CAP_PROP_FPS)
    frame_interval = int(video_fps / frame_rate)
    features = []
    timestamps = []
    current_frame = 0
    success, frame = video_capture.read()
    while success:
        if current_frame % frame_interval == 0:
            transformed_image = image_transform(frame).unsqueeze(0).to(device)
            with torch.no_grad():
                feature = resnet_model(transformed_image).cpu().numpy().squeeze()
                features.append(feature)
            timestamps.append(video_capture.get(cv2.CAP_PROP_POS_MSEC) / 1000)
        success, frame = video_capture.read()
        current_frame += 1
    video_capture.release()
    return np.array(features), np.array(timestamps)

def get_audio_features(video_file_path):
    video_clip = VideoFileClip(video_file_path)
    temp_audio_file = "temp_audio.wav"
    video_clip.audio.write_audiofile(temp_audio_file, verbose=False, logger=None)
    audio_examples = vggish_input.wavfile_to_examples(temp_audio_file)
    with torch.no_grad():
        audio_feature = audio_model.forward(torch.from_numpy(audio_examples).to(device))
    os.remove(temp_audio_file)
    return audio_feature.cpu().numpy()

def load_data(dataset_directory):
    training_videos = sorted(glob(os.path.join(dataset_directory, 'train', '*.mp4')))
    testing_videos = sorted(glob(os.path.join(dataset_directory, 'test', '*.mp4')))
    with open(os.path.join(dataset_directory, 'train_meta.json'), 'r') as file:
        metadata = json.load(file)
    return training_videos, testing_videos, metadata

def compare_video_episodes(video_paths, similarity_threshold=0.9):
    visual_sequences = []
    audio_sequences = []
    for video_path in tqdm(video_paths, desc="Extracting features"):
        visual_feature, _ = get_visual_features(video_path)
        audio_feature = get_audio_features(video_path)
        visual_sequences.append(visual_feature)
        audio_sequences.append(audio_feature)

    similarity_scores = []
    for i in range(len(visual_sequences)):
        for j in range(i+1, len(visual_sequences)):
            distance, _, _, _ = accelerated_dtw(audio_sequences[i], audio_sequences[j], dist='euclidean')
            similarity_scores.append(((i, j), distance))
    return similarity_scores

def group_segments(video_paths):
    predictions = {}
    for video_path in tqdm(video_paths, desc="Clustering"):
        visual_feature, timestamps = get_visual_features(video_path)
        clustering = DBSCAN(eps=0.5, min_samples=2).fit(visual_feature)
        cluster_labels = clustering.labels_

        label_counts = defaultdict(int)
        for label in cluster_labels:
            if label != -1:
                label_counts[label] += 1
        if not label_counts:
            predictions[os.path.basename(video_path)] = [0, 3]
            continue
        most_common_label = max(label_counts, key=label_counts.get)
        indices = np.where(cluster_labels == most_common_label)[0]
        start_time = timestamps[indices[0]]
        end_time = timestamps[indices[-1]]
        predictions[os.path.basename(video_path)] = [float(start_time), float(end_time)]
    return predictions

def write_predictions(predictions, output_file='predictions.json'):
    with open(output_file, 'w') as file:
        json.dump(predictions, file, indent=2)

def process_data(dataset_directory):
    _, test_videos, _ = load_data(dataset_directory)
    segment_predictions = group_segments(test_videos)
    write_predictions(segment_predictions)

if __name__ == "__main__":
    process_data("dataset")
