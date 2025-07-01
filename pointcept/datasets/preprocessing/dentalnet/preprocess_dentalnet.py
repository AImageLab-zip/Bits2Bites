import os
import json
import random
import argparse
import shutil
import pandas as pd
from pathlib import Path

# === START: LandmarkFilter class from landmarks_filter.py ===
import json
import numpy as np
from pathlib import Path
import argparse
from typing import Dict, List, Tuple, Optional

class LandmarkFilter:
    def __init__(self, score_threshold: float = 0.5, min_instance_points: int = 10):
        """
        Initialize the landmark filter.
        """
        self.score_threshold = score_threshold
        self.min_instance_points = min_instance_points

        
        # Define tooth type classification based on FDI numbering
        self.molar_fdi_numbers = {
            # Upper jaw molars: 16, 17, 18, 26, 27, 28
            16, 17, 18, 26, 27, 28,
            # Lower jaw molars: 36, 37, 38, 46, 47, 48
            36, 37, 38, 46, 47, 48
        }
        
        # Define minimum label requirements per tooth type
        self.molar_min_labels = {
            'Mesial': 1,
            'Distal': 1, 
            'FacialPoint': 1,
            'OuterPoint': 1,
            'Cusp': 2,  # Molars need at least 2 cusps
            'InnerPoint': 1
        }
        
        self.non_molar_min_labels = {
            'Mesial': 1,
            'Distal': 1,
            'FacialPoint': 1, 
            'OuterPoint': 1,
            'Cusp': 0,  # Non-molars have no cusps
            'InnerPoint': 1
        }
    
    def load_json(self, filepath: str) -> Dict:
        """Load JSON file safely."""
        try:
            with open(filepath, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            return {}
    
    def save_json(self, data: Dict, filepath: str) -> None:
        """Save JSON file safely."""
        try:
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
            print(f"Saved filtered data to {filepath}")
        except Exception as e:
            print(f"Error saving {filepath}: {e}")
    
    def get_valid_instances(self, instances: List[int], labels: List[int]) -> set:
        """
        Get valid tooth instances based on instance count and non-gingiva labels.
        Returns: Set of valid instance IDs
        """
        # Count vertices per instance
        instance_counts = {}
        for i, instance_id in enumerate(instances):
            if instance_id > 0:  # Skip gingiva (0) and invalid (-1)
                if instance_id not in instance_counts:
                    instance_counts[instance_id] = 0
                instance_counts[instance_id] += 1
        
        # Filter instances with sufficient vertices
        valid_instances = set()
        for instance_id, count in instance_counts.items():
            if count >= self.min_instance_points:
                valid_instances.add(instance_id)
        
        # Additional validation: check if instance has valid FDI labels
        instance_labels = {}
        for i, (instance_id, label) in enumerate(zip(instances, labels)):
            if instance_id > 0 and label > 0:  # Valid instance and non-gingiva label
                if instance_id not in instance_labels:
                    instance_labels[instance_id] = set()
                instance_labels[instance_id].add(label)
        
        # Keep only instances that have consistent, valid FDI labels
        final_valid_instances = set()
        for instance_id in valid_instances:
            if instance_id in instance_labels:
                # Check if instance has valid FDI tooth numbers (11-18, 21-28, 31-38, 41-48)
                valid_fdi_ranges = [
                    range(11, 19), range(21, 29),  # Upper jaw
                    range(31, 39), range(41, 49)   # Lower jaw
                ]
                
                instance_fdi_labels = instance_labels[instance_id]
                has_valid_fdi = any(
                    any(label in fdi_range for label in instance_fdi_labels)
                    for fdi_range in valid_fdi_ranges
                )
                
                if has_valid_fdi:
                    final_valid_instances.add(instance_id)
        
        return final_valid_instances
    
    def get_landmark_instance_mapping(self, landmarks_data: Dict, 
                                    instances: List[int], 
                                    mesh_vertices: Optional[List[List[float]]] = None) -> Dict[str, int]:
        """
        Map landmarks to their corresponding tooth instances based on spatial proximity.
        Returns: Dictionary mapping landmark uuid to instance_id
        """
        landmark_to_instance = {}
        
        for obj in landmarks_data.get('objects', []):
            landmark_uuid = obj.get('key', '')
            landmark_coord = obj.get('coord', [])
            instance_id = obj.get('instance_id', -1)
            
            # If instance_id is already provided and valid, use it
            if instance_id > 0:
                landmark_to_instance[landmark_uuid] = instance_id
            elif mesh_vertices and landmark_coord:
                # Find closest vertex and get its instance
                min_distance = float('inf')
                closest_instance = -1
                
                for i, vertex in enumerate(mesh_vertices):
                    if i < len(instances) and instances[i] > 0:
                        distance = np.sqrt(sum((a - b) ** 2 for a, b in zip(landmark_coord, vertex)))
                        if distance < min_distance:
                            min_distance = distance
                            closest_instance = instances[i]
                
                if closest_instance > 0:
                    landmark_to_instance[landmark_uuid] = closest_instance
        
        return landmark_to_instance
    
    def is_molar(self, fdi_number: int) -> bool:
        return fdi_number in self.molar_fdi_numbers
    
    def get_tooth_fdi_from_instance(self, instance_id: int, instances: List[int], labels: List[int]) -> int:
        fdi_labels = []
        for i, (inst_id, label) in enumerate(zip(instances, labels)):
            if inst_id == instance_id and label > 0:
                fdi_labels.append(label)
        
        if fdi_labels:
            # Return the most common FDI label for this instance
            from collections import Counter
            return Counter(fdi_labels).most_common(1)[0][0]
        return -1
    
    def get_minimum_labels_for_tooth(self, fdi_number: int) -> Dict[str, int]:
        """Get minimum label requirements for a specific tooth."""
        if self.is_molar(fdi_number):
            return self.molar_min_labels.copy()
        else:
            return self.non_molar_min_labels.copy()
    
    def apply_tooth_specific_filtering(self, landmarks_by_instance: Dict[int, List[Dict]], 
                                     instances: List[int], labels: List[int]) -> Dict[int, List[Dict]]:
        """
        Apply tooth-specific minimum label requirements.

        Returns: Filtered landmarks by instance
        """
        filtered_landmarks_by_instance = {}
        
        for instance_id, landmarks in landmarks_by_instance.items():
            # Get the FDI number for this tooth instance
            fdi_number = self.get_tooth_fdi_from_instance(instance_id, instances, labels)
            
            if fdi_number <= 0:
                print(f"Warning: Could not determine FDI number for instance {instance_id}")
                continue
            
            # Get minimum requirements for this tooth type
            min_requirements = self.get_minimum_labels_for_tooth(fdi_number)
            tooth_type = "molar" if self.is_molar(fdi_number) else "non-molar"
            
            print(f"Processing instance {instance_id} (FDI: {fdi_number}, type: {tooth_type})")
            
            # Group landmarks by class
            landmarks_by_class = {}
            for landmark in landmarks:
                landmark_class = landmark.get('class', '')
                if landmark_class not in landmarks_by_class:
                    landmarks_by_class[landmark_class] = []
                landmarks_by_class[landmark_class].append(landmark)
            
            # Sort landmarks within each class by score (highest first)
            for class_name in landmarks_by_class:
                landmarks_by_class[class_name].sort(
                    key=lambda x: x.get('score', 0.0), reverse=True
                )
            
            # Apply filtering logic for each class
            filtered_landmarks = []
            class_stats = {}
            
            for class_name, min_count in min_requirements.items():
                available_landmarks = landmarks_by_class.get(class_name, [])
                class_stats[class_name] = {
                    'available': len(available_landmarks),
                    'min_required': min_count,
                    'above_threshold': len([l for l in available_landmarks if l.get('score', 0) >= self.score_threshold])
                }
                
                if min_count == 0:
                    # For classes that should have 0 landmarks (e.g., Cusp for non-molars)
                    # Only keep landmarks that are significantly above threshold
                    high_score_landmarks = [l for l in available_landmarks 
                                          if l.get('score', 0) >= self.score_threshold]
                    filtered_landmarks.extend(high_score_landmarks)
                    class_stats[class_name]['kept'] = len(high_score_landmarks)
                    
                elif len(available_landmarks) == 0:
                    # No landmarks available for required class
                    class_stats[class_name]['kept'] = 0
                    print(f"  Warning: No {class_name} landmarks available for instance {instance_id}")
                    
                else:
                    # Determine how many to keep
                    above_threshold = [l for l in available_landmarks 
                                     if l.get('score', 0) >= self.score_threshold]
                    
                    if len(above_threshold) >= min_count:
                        # Enough high-quality landmarks available
                        filtered_landmarks.extend(above_threshold)
                        class_stats[class_name]['kept'] = len(above_threshold)
                    else:
                        # Not enough high-quality landmarks, take minimum required
                        # First take all above threshold, then fill with highest scoring below threshold
                        landmarks_to_take = above_threshold.copy()
                        remaining_needed = min_count - len(above_threshold)
                        
                        below_threshold = [l for l in available_landmarks 
                                         if l.get('score', 0) < self.score_threshold]
                        
                        # Take the highest scoring ones below threshold to meet minimum
                        landmarks_to_take.extend(below_threshold[:remaining_needed])
                        
                        filtered_landmarks.extend(landmarks_to_take)
                        class_stats[class_name]['kept'] = len(landmarks_to_take)
                        
                        if remaining_needed > 0:
                            print(f"  Info: Taking {remaining_needed} below-threshold {class_name} "
                                  f"landmarks for instance {instance_id} to meet minimum requirement")
            
            # Print statistics for this tooth
            print(f"  Class statistics for instance {instance_id}:")
            for class_name, stats in class_stats.items():
                if stats['min_required'] > 0 or stats['available'] > 0:
                    print(f"    {class_name}: {stats['kept']}/{stats['available']} kept "
                          f"(min: {stats['min_required']}, above_threshold: {stats['above_threshold']})")
            
            if filtered_landmarks:
                filtered_landmarks_by_instance[instance_id] = filtered_landmarks
            
        return filtered_landmarks_by_instance
    
    def filter_landmarks(self, landmarks_file: str, segmentation_file: str, 
                        output_landmarks_file: str, output_segmentation_file: str,
                        mesh_vertices: Optional[List[List[float]]] = None) -> Tuple[int, int]:
        """
        Filter landmarks and segmentation data to keep only valid points.
            
        Returns: Tuple of (original_landmark_count, filtered_landmark_count)
        """
        # Load data
        landmarks_data = self.load_json(landmarks_file)
        segmentation_data = self.load_json(segmentation_file)
        
        if not landmarks_data or not segmentation_data:
            print("Failed to load input files")
            return 0, 0
        
        instances = segmentation_data.get('instances', [])
        labels = segmentation_data.get('labels', [])
        
        if not instances or not labels:
            print("Invalid segmentation data")
            return 0, 0
        
        # Get valid tooth instances
        valid_instances = self.get_valid_instances(instances, labels)
        print(f"Found {len(valid_instances)} valid tooth instances: {sorted(valid_instances)}")
        
        # Map landmarks to instances
        landmark_to_instance = self.get_landmark_instance_mapping(
            landmarks_data, instances, mesh_vertices
        )
        
        # Filter landmarks
        original_landmarks = landmarks_data.get('objects', [])
        
        # Group landmarks by instance
        landmarks_by_instance = {}
        for landmark in original_landmarks:
            landmark_uuid = landmark.get('key', '')
            instance_id = landmark_to_instance.get(landmark_uuid, -1)
            
            if instance_id in valid_instances:
                if instance_id not in landmarks_by_instance:
                    landmarks_by_instance[instance_id] = []
                landmarks_by_instance[instance_id].append(landmark)
        
        print(f"Landmarks grouped by instance: {[(k, len(v)) for k, v in landmarks_by_instance.items()]}")
        
        # Apply tooth-specific filtering
        filtered_landmarks_by_instance = self.apply_tooth_specific_filtering(
            landmarks_by_instance, instances, labels
        )
        
        # Flatten filtered landmarks
        filtered_landmarks = []
        for instance_landmarks in filtered_landmarks_by_instance.values():
            for landmark in instance_landmarks:
                # Clean up the landmark data (remove score for final output)
                clean_landmark = {
                    'key': landmark.get('key', ''),
                    'class': landmark.get('class', ''),
                    'coord': landmark.get('coord', [])
                }
                filtered_landmarks.append(clean_landmark)
        
        # Create filtered landmarks output
        filtered_landmarks_data = {
            'version': landmarks_data.get('version', '1.0'),
            'description': landmarks_data.get('description', 'landmarks'),
            'key': landmarks_data.get('key', ''),
            'objects': filtered_landmarks
        }
        
        # Create filtered segmentation output (add back missing fields)
        # Extract patient ID and jaw from the key if possible
        key = landmarks_data.get('key', '')
        patient_id = key.split('/')[1].split('_')[0] if '/' in key and '_' in key else 'unknown'
        jaw = 'lower' if 'lower' in key.lower() else 'upper' if 'upper' in key.lower() else 'unknown'
        
        filtered_segmentation_data = {
            'id_patient': patient_id,
            'jaw': jaw,
            'instances': instances,
            'labels': labels
        }
        
        # Save filtered data
        self.save_json(filtered_landmarks_data, output_landmarks_file)
        self.save_json(filtered_segmentation_data, output_segmentation_file)
        
        original_count = len(original_landmarks)
        filtered_count = len(filtered_landmarks)
        
        print(f"\n=== Filtering Summary ===")
        print(f"Filtered landmarks: {original_count} -> {filtered_count} "
              f"(removed {original_count - filtered_count})")
        print(f"Score threshold: {self.score_threshold}")
        print(f"Valid instances used: {sorted(valid_instances)}")
        
        # Print detailed statistics
        total_above_threshold = sum(1 for landmark in original_landmarks 
                                   if landmark.get('score', 0) >= self.score_threshold)
        print(f"Original landmarks above threshold: {total_above_threshold}")
        print(f"Landmarks kept due to minimum requirements: {filtered_count - total_above_threshold}")
        
        return original_count, filtered_count
    
    def batch_filter(self, input_dir: str, output_dir: str, 
                    landmarks_suffix: str = '_landmarks.json', 
                    segmentation_suffix: str = '_segmentation.json') -> None:
        """
        Batch process multiple landmark and segmentation file pairs.
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Find all landmark files
        landmark_files = list(input_path.glob(f'*{landmarks_suffix}'))
        
        total_original = 0
        total_filtered = 0
        
        for landmark_file in landmark_files:
            # Find corresponding segmentation file
            base_name = landmark_file.stem.replace(landmarks_suffix.replace('.json', ''), '')
            segmentation_file = input_path / f"{base_name}{segmentation_suffix}"
            
            if not segmentation_file.exists():
                print(f"Warning: No segmentation file found for {landmark_file}")
                continue
            
            # Create output file names
            output_landmarks = output_path / f"{base_name}_filtered{landmarks_suffix}"
            output_segmentation = output_path / f"{base_name}_filtered{segmentation_suffix}"
            
            print(f"\nProcessing: {landmark_file.name}")
            orig, filt = self.filter_landmarks(
                str(landmark_file), str(segmentation_file),
                str(output_landmarks), str(output_segmentation)
            )
            
            total_original += orig
            total_filtered += filt
        
        print(f"\n=== Batch Processing Complete ===")
        print(f"Total landmarks processed: {total_original}")
        print(f"Total landmarks kept: {total_filtered}")
        print(f"Overall filtering ratio: {total_filtered/total_original*100:.1f}%" if total_original > 0 else "No landmarks processed")

def merge_upper_lower(input_dir, merged_output_dir):
    input_dir = Path(input_dir)
    merged_output_dir = Path(merged_output_dir)
    merged_output_dir.mkdir(parents=True, exist_ok=True)

    upper_files = sorted(input_dir.glob("*_upper_filtered_landmarks.json"))

    for upper_file in upper_files:
        base = upper_file.name.replace("_upper_filtered_landmarks.json", "")
        lower_file = input_dir / f"{base}_lower_filtered_landmarks.json"
        if not lower_file.exists():
            print(f"Skipping {base}: no matching lower file found")
            continue

        with open(upper_file) as f:
            upper_data = json.load(f)
        with open(lower_file) as f:
            lower_data = json.load(f)

        merged_data = {
            "version": "1.1",
            "description": "landmarks",
            "key": f"dental_{base}",
            "objects": upper_data["objects"] + lower_data["objects"]
        }

        with open(merged_output_dir / f"dental_{base}.json", "w") as out:
            json.dump(merged_data, out, indent=2)

def train_test_split(json_dir, output_dir, train_ratio=0.8):
    json_dir = Path(json_dir)
    output_dir = Path(output_dir)
    train_dir = output_dir / "train"
    test_dir = output_dir / "test"
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    all_jsons = list(json_dir.glob("*.json"))
    random.shuffle(all_jsons)
    split_idx = int(len(all_jsons) * train_ratio)

    for i, file in enumerate(all_jsons):
        dest = train_dir if i < split_idx else test_dir
        shutil.copy(file, dest / file.name)

def process_labels(xlsx_file, output_csv):
    df = pd.read_excel(xlsx_file, header=None)
    df.columns = df.iloc[0]
    df = df[1:].reset_index(drop=True)  # remove colums names row

    # Normalize class columns
    df["CLASSE DX"] = df["CLASSE DX"].replace({
        "Seconda Classe Testa a Testa": "Seconda Classe",
        "Seconda Classe Piena": "Seconda Classe"
    })
    df["CLASSE SX"] = df["CLASSE SX"].replace({
        "Seconda Classe Testa a Testa": "Seconda Classe",
        "Seconda Classe Piena": "Seconda Classe"
    })

    # Normalize morso anteriore
    df["MORSO ANTERIORE"] = df["MORSO ANTERIORE"].replace({
        "Morso Profondo": "Profondo",
        "Morso Aperto": "Aperto",
        "Morso Inverso": "Inverso"
    })

    # Drop TRASVERSALE and shift columns left
    df.drop(columns=["TRASVERSALE"], inplace=True)
    df.columns = [c for c in df.columns if pd.notna(c)]

    # Normalize TRASVERSALE (senza id denti)
    df["TRASVERSALE (senza id denti)"] = df["TRASVERSALE (senza id denti)"].replace({
        "Cross Bite": "Cross",
        "Scissor Bite": "Scissor",
        "Cross Bite / Scissor Bite": "Cross"
    })

    df.to_csv(output_csv, index=False)

def main():
    parser = argparse.ArgumentParser(description='Complete preprocessing pipeline')
    parser.add_argument('--input_dir', type=str, required=True, help='Input directory of landmark/segmentation files')
    parser.add_argument('--filtered_dir', type=str, required=True, help='Intermediate output for filtered files')
    parser.add_argument('--merged_dir', type=str, required=True, help='Output directory for merged landmark JSON files')
    parser.add_argument('--output_dir', type=str, required=True, help='Final output directory with train/test and labels')
    parser.add_argument('--xlsx_labels', type=str, required=True, help='Path to input Excel labels file')
    args = parser.parse_args()

    print("STEP 1: Filtering landmarks...")
    LandmarkFilter().batch_filter(args.input_dir, args.filtered_dir)

    print("STEP 2: Merging upper and lower...")
    merge_upper_lower(args.filtered_dir, args.merged_dir)

    print("STEP 3: Train/Test split...")
    train_test_split(args.merged_dir, args.output_dir)

    print("STEP 4: Processing labels...")
    process_labels(args.xlsx_labels, Path(args.output_dir) / "labels.csv")

    print("Preprocessing complete.")

if __name__ == "__main__":
    main()
