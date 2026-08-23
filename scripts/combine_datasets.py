import os
import shutil
import random

def combine_datasets():
    # Setup Paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.join(script_dir, "..")
    
    old_dataset_dir = os.path.join(project_dir, "dataset_balanced")
    new_dataset_dir = os.path.join(project_dir, "original")
    final_dataset_dir = os.path.join(project_dir, "dataset_final")

    # Target maximum images per category
    MAX_IMAGES = 1000

    # Ensure source directories exist
    if not os.path.exists(old_dataset_dir):
        print(f"Error: {old_dataset_dir} does not exist.")
        return
    if not os.path.exists(new_dataset_dir):
        print(f"Error: {new_dataset_dir} does not exist.")
        return

    # Create the final output directory
    if os.path.exists(final_dataset_dir):
        print(f"Cleaning up existing {final_dataset_dir}...")
        shutil.rmtree(final_dataset_dir)
    os.makedirs(final_dataset_dir)

    # Define the mapping
    # Format: "target_category": ( [old_dataset_folders], [new_dataset_folders] )
    mappings = {
        "wet_bio_degradable": (
            ["wet_bio_degradable"],
            ["biological"]
        ),
        "plastic": (
            ["plastic"],
            ["plastic"]
        ),
        "paper_and_cardboard": (
            ["paper_and_cardboard"],
            ["paper", "cardboard"]
        ),
        "metal": (
            ["metal"],
            ["metal"]
        ),
        "reject_other": (
            ["reject_other"],
            ["trash", "glass", "battery", "clothes", "shoes"]
        )
    }

    random.seed(42) # For reproducibility

    for target_cat, (old_folders, new_folders) in mappings.items():
        all_images = []

        # 1. Collect images from Old Dataset
        for folder in old_folders:
            folder_path = os.path.join(old_dataset_dir, folder)
            if os.path.exists(folder_path):
                images = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.jfif'))]
                for img in images:
                    all_images.append((folder_path, img))

        # 2. Collect images from New Dataset
        for folder in new_folders:
            folder_path = os.path.join(new_dataset_dir, folder)
            if os.path.exists(folder_path):
                images = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.jfif'))]
                for img in images:
                    all_images.append((folder_path, img))

        # 3. Shuffle the collected images
        random.shuffle(all_images)

        # 4. Cap at 1000 images
        if len(all_images) > MAX_IMAGES:
            final_images = all_images[:MAX_IMAGES]
        else:
            final_images = all_images
            print(f"Note: '{target_cat}' only has {len(all_images)} total images (Target was {MAX_IMAGES}).")

        # 5. Create Target Directory and Copy Images
        target_dir_path = os.path.join(final_dataset_dir, target_cat)
        os.makedirs(target_dir_path, exist_ok=True)

        print(f"Copying {len(final_images)} images into '{target_cat}'...")
        for count, (src_folder, img_name) in enumerate(final_images):
            src_path = os.path.join(src_folder, img_name)
            # Create a unique filename to prevent overwriting if both datasets have an image named 'image_1.jpg'
            new_img_name = f"{target_cat}_{count:04d}_{img_name}"
            dest_path = os.path.join(target_dir_path, new_img_name)
            shutil.copy2(src_path, dest_path)

    print("\n✅ Dataset combination complete! Check the 'dataset_final' folder.")

if __name__ == "__main__":
    combine_datasets()
