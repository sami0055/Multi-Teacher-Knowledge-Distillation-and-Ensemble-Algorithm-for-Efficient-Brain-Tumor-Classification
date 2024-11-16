def apply_unsharp_mask(image):
    # Convert the image to an array and ensure it's in the correct range for processing
    image_np = img_to_array(image)
    
    # Apply a stronger Gaussian blur
    blurred = cv2.GaussianBlur(image_np, (7, 7), sigmaX=2.0)
    
    # Increase the sharpening intensity in the unsharp mask formula
    unsharp_image = cv2.addWeighted(image_np, 2.0, blurred, -1.0, 0)
    
    # Clip values to maintain valid pixel range
    unsharp_image = np.clip(unsharp_image, 0, 255)
    return unsharp_image.astype(np.uint8)


# Load and test the sample image
sample_image_path = '/content/drive/MyDrive/Brain Tumor Classification/Merged_Dataset_1/Training/glioma/Tr-gl_0010.jpg'
sample_image = load_img(sample_image_path, target_size=(160, 160))

# Apply the revised unsharp mask
sharpened_image = apply_unsharp_mask(sample_image)

# Display original and sharpened images for comparison
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(sample_image)
plt.title("Original Image")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(array_to_img(sharpened_image))
plt.title("Sharpened Image ")
plt.axis('off')

plt.show()


def preprocess_and_save_images(input_dir, output_dir, target_size=(160, 160)):
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Iterate through all subdirectories and images
    for class_dir in os.listdir(input_dir):
        class_path = os.path.join(input_dir, class_dir)
        if not os.path.isdir(class_path):
            continue
            
        # Create a corresponding class directory in output directory
        output_class_path = os.path.join(output_dir, class_dir)
        os.makedirs(output_class_path, exist_ok=True)
        
        for img_file in os.listdir(class_path):
            img_path = os.path.join(class_path, img_file)
            
            # Load and preprocess image
            image = load_img(img_path, target_size=target_size)
            unsharp_image = apply_unsharp_mask(image)
            
            # Convert array back to image and save
            processed_img = array_to_img(unsharp_image)
            processed_img.save(os.path.join(output_class_path, img_file))

# Define paths for saving preprocessed images
processed_train_path = '/content/drive/MyDrive/Brain Tumor Classification/Preprocessed_Training'
processed_test_path = '/content/drive/MyDrive/Brain Tumor Classification/Preprocessed_Testing'

# Preprocess and save images
preprocess_and_save_images(train_path, processed_train_path, target_size=(160, 160))
preprocess_and_save_images(test_path, processed_test_path, target_size=(160, 160))
