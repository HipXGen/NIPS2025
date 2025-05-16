# run_paper_benchmark.py
import torch
import torchvision.models as models
from torchvision import transforms
from torch.utils.data import Dataset as PyTorchDataset 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import numpy as np
import os
from tqdm import tqdm
import argparse
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(module)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

class AllInOnePTDataset(PyTorchDataset):
    def __init__(self, pt_files_dir, target_label_key, 
                 image_key='image', text_features_key='text_for_tfidf',
                 expected_image_count=None, resolution=512):
        self.pt_files_dir = pt_files_dir
        self.target_label_key = target_label_key
        self.image_key = image_key
        self.text_features_key = text_features_key
        self.resolution = resolution
        
        logger.info(f"Initializing AllInOnePTDataset...")
        logger.info(f"Scanning for .pt files in: {pt_files_dir}")

        if not os.path.isdir(pt_files_dir):
            raise ValueError(f"PT files directory '{pt_files_dir}' not found.")

        self.file_paths = sorted([
            os.path.join(pt_files_dir, f)
            for f in os.listdir(pt_files_dir) if f.lower().endswith('.pt')
        ])
        
        if not self.file_paths:
            raise ValueError(f"No .pt files found in '{pt_files_dir}'.")
        
        logger.info(f"Found {len(self.file_paths)} .pt files.")
        if expected_image_count is not None and len(self.file_paths) != expected_image_count:
            logger.warning(f"Expected {expected_image_count} .pt files, but found {len(self.file_paths)}.")

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        pt_path = self.file_paths[idx]
        try:
            loaded_dict = torch.load(pt_path, map_location='cpu')
            
            image_tensor = loaded_dict.get(self.image_key)
            text_features_tensor = loaded_dict.get(self.text_features_key)
            target_label = loaded_dict.get(self.target_label_key)
            image_id = loaded_dict.get('original_filename', os.path.splitext(os.path.basename(pt_path))[0])


            if image_tensor is None: raise ValueError(f"'{self.image_key}' not found or is None in {pt_path}")
            if text_features_tensor is None: raise ValueError(f"'{self.text_features_key}' not found or is None in {pt_path}")
            if target_label is None: raise ValueError(f"Target label '{self.target_label_key}' not found in {pt_path}")

            if not (isinstance(image_tensor, torch.Tensor) and image_tensor.shape == torch.Size([3, self.resolution, self.resolution])):
                raise ValueError(f"Image tensor from {pt_path} unexpected format. Shape: {image_tensor.shape}")
            if not (isinstance(text_features_tensor, torch.Tensor) and len(text_features_tensor.shape) == 1):
                if len(text_features_tensor.shape) == 2 and text_features_tensor.shape[0] == 1:
                    text_features_tensor = text_features_tensor.squeeze(0)
                else:
                    raise ValueError(f"Text feature tensor from {pt_path} unexpected shape: {text_features_tensor.shape}")
                
            return {
                "image": image_tensor.float(), # Ensure float
                "text_features": text_features_tensor.float(), # Ensure float
                "label": target_label, # This will be numerically encoded later
                "image_id": image_id
            }
        except Exception as e:
            logger.error(f"Error in __getitem__ for pt: {pt_path}: {e}")
            raise e

def run_benchmark(pt_data_dir_arg, expected_images_arg, target_label_key_arg, 
                  image_key_in_pt_arg='image', text_features_key_in_pt_arg='text_for_tfidf', resolution_arg=512):
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {DEVICE}")

    full_dataset = AllInOnePTDataset(
        pt_files_dir=pt_data_dir_arg,
        target_label_key=target_label_key_arg,
        image_key=image_key_in_pt_arg,
        text_features_key=text_features_key_in_pt_arg,
        expected_image_count=expected_images_arg,
        resolution=resolution_arg
    )

    if len(full_dataset) == 0: 
        logger.info("Dataset is empty. Benchmark cannot run.")
        return

    all_image_tensors_for_resnet = []
    all_text_tfidf_features_np = []
    all_target_labels_str_or_int = []

    logger.info("Collecting features and labels from all samples...")
    for i in tqdm(range(len(full_dataset)), desc="Loading Samples for Benchmark"):
        try:
            sample = full_dataset[i]
            all_image_tensors_for_resnet.append(sample["image"])
            all_text_tfidf_features_np.append(sample["text_features"].numpy())
            all_target_labels_str_or_int.append(sample["label"])
        except Exception as e:
            logger.warning(f"Skipping sample at index {i} due to error in __getitem__: {e}")
            continue
    
    if not all_target_labels_str_or_int:
        logger.error("No samples with labels collected. Aborting benchmark.")
        return
    logger.info(f"Collected data for {len(all_target_labels_str_or_int)} samples for benchmark.")

    logger.info("Extracting image features using ResNet50...")
    resnet50 = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    resnet50.fc = torch.nn.Identity()
    resnet50.eval().to(DEVICE)
    imagenet_normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    image_cnn_features_list = []
    temp_image_dataset_for_resnet = torch.utils.data.TensorDataset(torch.stack(all_image_tensors_for_resnet))
    temp_image_loader_for_resnet = torch.utils.data.DataLoader(temp_image_dataset_for_resnet, batch_size=32, shuffle=False)

    for (batch_img_tensors,) in tqdm(temp_image_loader_for_resnet, desc="Image Feature Extraction"):
        img_tensor_neg1_1 = batch_img_tensors
        img_tensor_0_1 = (img_tensor_neg1_1 * 0.5) + 0.5
        img_tensor_imagenet_norm = imagenet_normalizer(img_tensor_0_1)
        with torch.no_grad():
            img_feat_batch = resnet50(img_tensor_imagenet_norm.to(DEVICE))
        image_cnn_features_list.extend(img_feat_batch.cpu().numpy())
    image_cnn_features = np.array(image_cnn_features_list)
    logger.info(f"Image CNN feature shape: {image_cnn_features.shape}")
    
    all_text_tfidf_features_np = np.array(all_text_tfidf_features_np)
    logger.info(f"Text TF-IDF feature shape: {all_text_tfidf_features_np.shape}")

    le = LabelEncoder()
    all_labels_numeric = le.fit_transform(all_target_labels_str_or_int)
    logger.info(f"Target label classes: {list(le.classes_)} (encoded to 0-{len(le.classes_)-1})")

    if len(np.unique(all_labels_numeric)) < 2:
        logger.error(f"Not enough unique classes in target label for classification. Need at least 2.")
        return

    feature_sets = {}
    if all_text_tfidf_features_np.size > 0 and all_text_tfidf_features_np.shape[0] == len(all_labels_numeric):
        feature_sets["Text-Only (TF-IDF)"] = all_text_tfidf_features_np
    if image_cnn_features.size > 0 and image_cnn_features.shape[0] == len(all_labels_numeric):
        feature_sets["Image-Only (ResNet50)"] = image_cnn_features
    if "Text-Only (TF-IDF)" in feature_sets and "Image-Only (ResNet50)" in feature_sets:
         feature_sets["Multi-Modal (Image + TF-IDF)"] = np.concatenate((image_cnn_features, all_text_tfidf_features_np), axis=1)
    
    for fs_name, X_features in feature_sets.items():
        logger.info(f"\n--- Running Benchmark for: {fs_name} ---")
        logger.info(f"Feature shape: {X_features.shape}")

        X_train, X_test, y_train, y_test = train_test_split(
            X_features, all_labels_numeric, test_size=0.25, random_state=42, stratify=all_labels_numeric
        )
        
        scaler = StandardScaler(with_mean=(not ("Text-Only" in fs_name and "Image" not in fs_name))) 
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        classifier = LogisticRegression(max_iter=1000, solver='liblinear', class_weight='balanced')
        classifier.fit(X_train_scaled, y_train)
        y_pred = classifier.predict(X_test_scaled)

        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, target_names=[str(c) for c in le.classes_], zero_division=0)
        print(f"\n--- Results for {fs_name} (Target: {target_label_key_arg}) ---")
        print(f"Accuracy: {accuracy:.4f}")
        print("Classification Report:\n" + report)
        logger.info(f"Results for {fs_name}:\nAccuracy: {accuracy:.4f}\nClassification Report:\n{report}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run benchmark experiments using self-contained .pt files.")
    parser.add_argument("--pt_data_dir", type=str, default="/mnt/workspace/processed_xray_data", help="Directory with .pt files (dictionaries).")
    parser.add_argument("--expected_images", type=int, default=1250, help="Expected number of images.")
    parser.add_argument("--target_label_key", type=str, default="diagnosis_label", help="Key in the .pt dict for the target label.")
    parser.add_argument("--image_key", type=str, default="image", help="Key for image tensor in .pt dict.")
    parser.add_argument("--text_features_key", type=str, default="text_for_tfidf", help="Key for TF-IDF tensor in .pt dict.")
    parser.add_argument("--resolution", type=int, default=512, help="Image resolution for Dataset class validation.")
    
    cli_args = parser.parse_args()
    
    run_benchmark(
        pt_data_dir_arg=cli_args.pt_data_dir, 
        expected_images_arg=cli_args.expected_images,
        target_label_key_arg=cli_args.target_label_key,
        text_features_key_in_pt_arg=cli_args.text_features_key,
        resolution_arg=cli_args.resolution
    )
