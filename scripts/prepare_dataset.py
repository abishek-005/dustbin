import os
import shutil
import random

def prepare_dataset(source_dir, dest_dir, samples_per_category=200):
    if not os.path.exists(source_dir):
        print(f"Error: Source directory '{source_dir}' does not exist.")
        return

    # Define the new categories
    categories = [
        "wet_bio_degradable",
        "plastic",
        "paper_and_cardboard",
        "metal",
        "reject_other"
    ]

    # Create destination directories
    for cat in categories:
        os.makedirs(os.path.join(dest_dir, cat), exist_ok=True)
        print(f"Created/Verified directory: {cat}")

    # The wet_bio_degradable is left empty as requested.

    # Mapping source folders to target categories
    mappings = {
        "plastic": ["plastic"],
        "paper_and_cardboard": ["paper", "cardboard"],
        "metal": ["metal"],
        "reject_other": ["trash", "glass"]
    }

    random.seed(42) # For reproducibility

    for target_cat, source_folders in mappings.items():
        all_images = []
        # Collect all images from the mapped source folders
        for folder in source_folders:
            folder_path = os.path.join(source_dir, folder)
            if os.path.exists(folder_path):
                images = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                for img in images:
                    all_images.append((folder_path, img))
            else:
                print(f"Warning: Source folder '{folder}' not found in '{source_dir}'.")

        # Sample 200 images
        if len(all_images) < samples_per_category:
            print(f"Warning: Not enough images for '{target_cat}'. Found {len(all_images)}, needed {samples_per_category}. Copying all.")
            sampled_images = all_images
        else:
            sampled_images = random.sample(all_images, samples_per_category)

        # Copy images
        print(f"Copying {len(sampled_images)} images to '{target_cat}'...")
        for i, (folder_path, img_name) in enumerate(sampled_images):
            src_path = os.path.join(folder_path, img_name)
            # Prefix with original folder name to avoid name collisions
            original_folder = os.path.basename(folder_path)
            dest_name = f"{original_folder}_{img_name}"
            dest_path = os.path.join(dest_dir, target_cat, dest_name)
            shutil.copy2(src_path, dest_path)
    
    print("Dataset preparation complete!")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Prepare balanced dataset")
    parser.add_argument("--source", type=str, required=True, help="Path to raw TrashType_Image_Dataset")
    parser.add_argument("--dest", type=str, default="../dataset_balanced", help="Destination path for balanced dataset")
    
    args = parser.parse_args()
    
    # Resolve absolute paths
    source_abs = os.path.abspath(args.source)
    dest_abs = os.path.abspath(args.dest)
    
    prepare_dataset(source_abs, dest_abs)
