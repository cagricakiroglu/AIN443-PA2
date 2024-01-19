import cv2
import numpy as np
import math
from matplotlib import pyplot as plt
import os
from sklearn.utils import resample

from sklearn.svm import SVC




def calculate_circle_overlap(circle1, circle2):
    x1, y1, r1 = circle1
    x2, y2, r2 = circle2
    distance = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
    return distance < (r1 + r2)


def merge_significantly_overlapping_circles(circles):
    groups = []
    used = set()

    # Group overlapping circles
    for i, circle1 in enumerate(circles):
        if i in used:
            continue
        group = [circle1]
        for j, circle2 in enumerate(circles):
            if j in used or j == i:
                continue
            if calculate_circle_overlap(circle1, circle2):
                group.append(circle2)
                used.add(j)
        groups.append(group)

    # Merge groups into single circles
    merged_circles = []
    for group in groups:
        if len(group) > 1:
            avg_x = int(sum(x for x, _, _ in group) / len(group))
            avg_y = int(sum(y for _, y, _ in group) / len(group))
            max_r = max(r for _, _, r in group)
            merged_circles.append((avg_x, avg_y, max_r))
        else:
            merged_circles.extend(group)

    return merged_circles

def resize_image(image, size):
    # Calculate the ratio of the new size and find the best match for the new dimensions
    # maintaining the aspect ratio.
    h, w = image.shape[:2]
    (width, height) = size

    # Calculate the ratio of the width and construct the dimensions
    if w > h:
        aspect = width / float(w)
        dim = (width, int(h * aspect))
    else:
        aspect = height / float(h)
        dim = (int(w * aspect), height)

    # Resize the image
    resized_image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

    # If the new size is smaller, then we need to pad the image
    # If the new size is larger, we need to crop the image
    delta_w = width - resized_image.shape[1]
    delta_h = height - resized_image.shape[0]
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    # Create a border around the image to maintain the aspect ratio
    color = [0, 0, 0]  # 'color' can be changed depending on the application
    new_resized_image = cv2.copyMakeBorder(resized_image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

    return new_resized_image

has_plotted_sobel = False

def hough_transform_circle_detection(image, radius_range, threshold, output_path):
    # Step 1: Preprocessing
    global has_plotted_sobel

    
    diameter = 3  # Diameter of each pixel neighborhood
    sigmaColor = 100  # Filter sigma in the color space
    sigmaSpace = 100  # Filter sigma in the coordinate space
    filtered_image = cv2.bilateralFilter(image, diameter, sigmaColor, sigmaSpace)


    smoothed_image = cv2.GaussianBlur(filtered_image, (3, 3), 2)
    edges = cv2.Canny(smoothed_image, 30, 220)

    # Compute the gradient direction for each pixel
    sobelx = cv2.Sobel(smoothed_image, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(smoothed_image, cv2.CV_64F, 0, 1, ksize=5)
    direction = np.arctan2(sobely, sobelx)
    magnitude = np.sqrt(sobelx**2 + sobely**2)
    
    if not has_plotted_sobel:
        plt.imshow(edges, cmap='gray')
        plt.title("Canny Edges")
        #plt.show()
        has_plotted_sobel = True  # Update the flag
        

    # Step 3: Initialize Hough Space
    hough_space = np.zeros((edges.shape[0], edges.shape[1], len(radius_range)))

    # Step 4: Cast Votes in Hough Space
    for x in range(edges.shape[1]):
        for y in range(edges.shape[0]):
            if edges[y, x] > 0:  # Check for edge
                for radius_index, radius in enumerate(radius_range):
                    a = int(x - radius * math.cos(direction[y, x]))
                    b = int(y - radius * math.sin(direction[y, x]))
                    if a >= 0 and a < edges.shape[1] and b >= 0 and b < edges.shape[0]:
                        hough_space[b, a, radius_index] += 1
     # Step 5: Find Local Maxima
    circles = []
    for radius_index, radius in enumerate(radius_range):
        # Find peaks in the Hough Space slice for the current radius
        peaks = np.where(hough_space[:, :, radius_index] > threshold)
        circles.extend([(x, y, radius) for y, x in zip(*peaks)])

    circles = merge_significantly_overlapping_circles(circles)


    # Draw Circles on the output image
    output_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    for x, y, radius in circles:
        cv2.circle(output_image, (x, y), radius, (0, 255, 0), 2)
        cv2.circle(output_image, (x, y), 2, (0, 0, 255), 3)  # center of the circle

    # Save the result image
    cv2.imwrite(output_path, output_image)

    return circles


def process_directory_and_save_circles_and_hold_resized_images(directory_path, size, radius_range, threshold,output):


    if not os.path.exists(output):
        os.makedirs(output)


    circles_per_image = {}  # Dictionary to store circles for each image
    resized_images = {} # Dictionary to store resized images
    only_resized_images ={}
 # Dictionary to store circles for each image

    for filename in os.listdir(directory_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
         
            image_path = os.path.join(directory_path, filename)
            
            image = cv2.imread(image_path)
            resized_image = resize_image(image, size)
            gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
            
          

             # Define the output path for the processed image
            output_path = os.path.join(output, f'processed_{filename}')
            only_resized_images[filename] = resized_image
            resized_images[filename] = gray_image
            
            


            circles = hough_transform_circle_detection(gray_image, radius_range, threshold,output_path)
            circles_per_image[filename] = circles

           

    return only_resized_images,circles_per_image, resized_images



# Example usage:
train_directory = 'src/Train'
testr_directory = 'src/TestR'
testv_directory = 'src/TestV'

# Define the parameters for Hough transform
train_radius_range = np.arange(10, 130, 2)
others_radius_range =np.arange(5 ,200, 2)   # Example radius range
threshold = 19 # Example threshold, this needs to be tuned to your specific application

# Process each directory
output_train="src/Train_Hough"
output_testr="src/TestR_Hough"
output_testv="src/TestV_Hough"


only_resize_train,train_circles, train_resized_images = process_directory_and_save_circles_and_hold_resized_images(train_directory, (200, 200), train_radius_range, threshold,output_train)
only_resize_testr,testr_circles, testr_resized_images = process_directory_and_save_circles_and_hold_resized_images(testr_directory, (1000, 1000), others_radius_range, threshold,output_testr)
only_resize_testv,testv_circles, testv_resized_images = process_directory_and_save_circles_and_hold_resized_images(testv_directory, (1000, 1000), others_radius_range, threshold,output_testv)




#print(testr_resized_images)

#print("done")

def collect_image_filenames(image_folder):
    # List comprehension to collect image filenames
    filenames = [file for file in os.listdir(image_folder) 
                 if os.path.isfile(os.path.join(image_folder, file)) 
                 and file.lower().endswith(('.jpg', '.png', '.jpeg', '.gif', '.bmp'))]
    return filenames


train_filenames = collect_image_filenames(train_directory)
testr_filenames = collect_image_filenames(testr_directory)
testv_filenames = collect_image_filenames(testv_directory)

def extract_roi(image, circle):
    h, w = image.shape[:2]  # Image dimensions
    x, y, r = circle

    # Ensure the circle's center and radius are within the image dimensions
    x, y, r = max(0, min(x, w)), max(0, min(y, h)), max(0, min(r, h, w))

    # Create a mask for the circle
    mask = np.zeros_like(image)
    cv2.circle(mask, (x, y), r, (255, 255, 255), -1)

    # Bitwise-and for region extraction
    roi = cv2.bitwise_and(image, mask)

    # Calculate cropping coordinates, ensuring they are within image bounds
    y1, y2 = max(0, y - r), min(h, y + r)
    x1, x2 = max(0, x - r), min(w, x + r)

    # Crop the ROI
    cropped_roi = roi[y1:y2, x1:x2]

    # Check if the ROI is empty
    if cropped_roi.size == 0:
        #print("Empty ROI detected. Image or circle parameters may be incorrect.")
        #print("Image shape:", image.shape)
        #print("Circle parameters:", circle)
        return None

    # Resize cropped ROI to 200x200 pixels
    resized_roi = cv2.resize(cropped_roi, (200, 200))

    return resized_roi





def extract_rois_from_resized_images(resized_images, circles_per_image):
    rois_per_image = {}  # Dictionary to store RoIs for each image

    for filename, image in resized_images.items():
        if filename in circles_per_image:
            rois = [extract_roi(image, circle) for circle in circles_per_image[filename]]
            rois_per_image[filename] = rois

    return rois_per_image

# Use the function to extract RoIs from the resized images
train_rois = extract_rois_from_resized_images(train_resized_images, train_circles)
testr_rois = extract_rois_from_resized_images(testr_resized_images, testr_circles)
testv_rois = extract_rois_from_resized_images(testv_resized_images, testv_circles)

#print("RoI extraction complete for all image sets")


def display_train_rois(train_rois, num_images=2, num_rois_per_image=20):
    fig, axs = plt.subplots(num_images, num_rois_per_image, figsize=(15, 3 * num_images))

    for i, (filename, rois) in enumerate(train_rois.items()):
        if i >= num_images:
            break
        for j in range(min(num_rois_per_image, len(rois))):
            axs[i, j].imshow(rois[j], cmap='gray')
            axs[i, j].set_title(f"{filename} - ROI {j+1}")
            axs[i, j].axis('off')

    #plt.tight_layout()
    #plt.show()

# Display some of the train RoIs
display_train_rois(testv_rois)

def compute_gradients(image):
    # Calculate gradients using Sobel operator
    gradient_values_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)
    gradient_values_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)
    
    # Calculate gradient magnitude and direction (in degrees)
    magnitude = np.sqrt(gradient_values_x**2 + gradient_values_y**2)
    angle = np.arctan2(gradient_values_y, gradient_values_x) * (180 / np.pi) % 180
    
    return magnitude, angle


def compute_gradients_for_rois(rois_per_image):
    gradients_per_image = {}
    
    for filename, rois in rois_per_image.items():
        gradients_per_roi = []
        for roi in rois:
            if roi.size == 0:  # Check if the ROI is empty
                #print(f"Empty ROI found in image {filename}. Skipping...")
                continue
            magnitude, angle = compute_gradients(roi)
            gradients_per_roi.append((magnitude, angle))
        gradients_per_image[filename] = gradients_per_roi
        
    return gradients_per_image

train_gradients = compute_gradients_for_rois(train_rois)
testr_gradients = compute_gradients_for_rois(testr_rois)
testv_gradients = compute_gradients_for_rois(testv_rois)

#print("Gradients computed for all RoIs")


def cell_histogram(magnitude, angle, cell_size, bin_size):
    # Initialize histogram
    num_bins = int(180 / bin_size)
    histogram = np.zeros(num_bins)
    
    # Divide the image into cells and calculate histogram for each cell
    for i in range(0, magnitude.shape[0], cell_size):
        for j in range(0, magnitude.shape[1], cell_size):
            cell_magnitude = magnitude[i:i+cell_size, j:j+cell_size]
            cell_angle = angle[i:i+cell_size, j:j+cell_size]
            
            # Discretize each angle into a bin
            bin_indices = (cell_angle // bin_size).astype(int)
            
            for m in range(cell_magnitude.shape[0]):
                for n in range(cell_magnitude.shape[1]):
                    bin_idx = bin_indices[m, n] % num_bins
                    histogram[bin_idx] += cell_magnitude[m, n]
    
    return histogram

def compute_histograms_for_gradients(gradients_per_image, cell_size=8, bin_size=20):
    histograms_per_image = {}

    for filename, gradients_list in gradients_per_image.items():
        histograms_per_roi = []
        for magnitude, angle in gradients_list:
            # Compute the histogram for each ROI
            histogram = cell_histogram(magnitude, angle, cell_size, bin_size)
            histograms_per_roi.append(histogram)
        histograms_per_image[filename] = histograms_per_roi
    
    return histograms_per_image


# Compute the HOG cell histograms for all ROIs
train_histograms = compute_histograms_for_gradients(train_gradients)
testr_histograms = compute_histograms_for_gradients(testr_gradients)
testv_histograms = compute_histograms_for_gradients(testv_gradients)

#print("HOG histograms computed for all RoIs")




def normalize_histogram(histogram):
    # L2-norm normalization
    l2_norm = np.sqrt(np.sum(np.square(histogram)))
    normalized_histogram = histogram / l2_norm if l2_norm > 0 else histogram
    return normalized_histogram

def normalize_histograms_list(histograms_list):
    normalized_histograms = [normalize_histogram(histogram) for histogram in histograms_list]
    return normalized_histograms
    
def normalize_histograms_per_image(histograms_per_image):
    normalized_histograms_per_image = {}

    for filename, histograms_list in histograms_per_image.items():
        normalized_histograms = normalize_histograms_list(histograms_list)
        normalized_histograms_per_image[filename] = normalized_histograms
    
    return normalized_histograms_per_image


# Normalize the HOG histograms for all ROIs
train_normalized_histograms = normalize_histograms_per_image(train_histograms)
testr_normalized_histograms = normalize_histograms_per_image(testr_histograms)
testv_normalized_histograms = normalize_histograms_per_image(testv_histograms)

#print("HOG histograms normalized for all RoIs")


def label_histograms(normalized_histograms_per_image, filenames, circle_points):
    labeled_histograms = []
    labels = []
    count_undetected = 0

    for filename in filenames:
        # Skip the image if no circles were detected
        if len(circle_points[filename]) == 0:
            count_undetected += 1
            continue
        
        # Check if there are histograms for this file
        if not normalized_histograms_per_image[filename]:
            #print(f"No histograms found for {filename}. Skipping...")
            continue

        # Extract the label from the filename
        lst = filename.split("_")
        cat = lst[0] + "_" + lst[1]
        
        # Assume each file has one histogram (adjust if there are multiple)
        histogram = normalized_histograms_per_image[filename][0]
        labeled_histograms.append(histogram)
        labels.append(cat)

    #print("Number of undetected images: ", count_undetected, "out of ", len(filenames))
    return labeled_histograms, labels

# Label the normalized histograms for the train data
train_labeled_histograms, train_labels = label_histograms(train_normalized_histograms, train_filenames, train_circles)

train_histograms_flat = []
train_labels_flat = []

for filename, histograms in train_normalized_histograms.items():
    if len(train_circles[filename]) == 0:  # Skip if no circles were detected
        continue

    # Extract label from filename
    lst = filename.split("_")
    cat = lst[0] + "_" + lst[1]

    # Flatten histograms and add them to the list with their labels
    for histogram in histograms:
        train_histograms_flat.append(histogram) # Ensure histograms are flattened
        train_labels_flat.append(cat)

# Convert the lists to NumPy arrays
train_histograms_array = np.array(train_histograms_flat)
train_labels_array = np.array(train_labels_flat)



# Assuming train_histograms_array and train_labels_array are defined
# Example:
# train_histograms_array = np.array([...])
# train_labels_array = np.array([...])

# Find the minimum frequency among the labels
unique, counts = np.unique(train_labels_array, return_counts=True)
min_frequency = min(counts)

# Initialize lists to store the downsampled data
downsampled_histograms = []
downsampled_labels = []

# Downsample each class to the minimum frequency
for label in unique:
    # Get the indices of the current label
    indices = np.where(train_labels_array == label)[0]
    
    # Downsample the indices
    downsampled_indices = resample(indices, replace=False, n_samples=min_frequency, random_state=0)
    
    # Append the downsampled histograms and labels to the lists
    downsampled_histograms.extend(train_histograms_array[downsampled_indices])
    downsampled_labels.extend(train_labels_array[downsampled_indices])

# Convert the lists back to numpy arrays
downsampled_histograms_array = np.array(downsampled_histograms)
downsampled_labels_array = np.array(downsampled_labels)

# Now downsampled_histograms_array and downsampled_labels_array are balanced

# Flatten the histograms and extract them along with their labels into lists


# Now you can fit the SVM classifier


svm_classifier = SVC(kernel='linear')



svm_classifier.fit(downsampled_histograms_array, downsampled_labels_array)



def classify_and_draw_circles(image, circles, classifier):
    for x, y, r in circles:
        
        # Extract the region of interest
        roi = image[y-r:y+r, x-r:x+r]
        #resized_roi = cv2.resize(roi, (256,256))  # Resize to match training data,
        magnitude, angle = compute_gradients(roi)
        histogram = cell_histogram(magnitude,angle,8,20)
        n_hist=normalize_histogram(histogram)

        # Predict the class (coin type)
        coin_type = classifier.predict([n_hist])[0]
  

        # Draw the circle and label
        cv2.circle(image, (x, y), r, (0, 255, 0), 2)  # Green circle
        cv2.putText(image, coin_type, (x - r, y - r - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)  # Blue text

    return image




def process_and_save_images(image_dict, circles_dict, classifier, folder_name):
    # Create the folder if it doesn't exist
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    num_images = len(image_dict)

    # Process each image and save the plot
    for i, (filename, image) in enumerate(image_dict.items()):
        if filename not in circles_dict:
            continue

        circles = circles_dict[filename]  # Get the circles for the current image
        image_with_circles = classify_and_draw_circles(image, circles, classifier)

        # Convert the image to RGB color format (assuming it's in BGR format)
        image_rgb = cv2.cvtColor(image_with_circles, cv2.COLOR_BGR2RGB)

        # Create a figure for each image
        plt.figure(figsize=(16, 16))  # Adjust the figsize as needed

        # Display the image with circles and labels, specifying RGB colormap
        plt.imshow(image_rgb, cmap=None)  # Specify cmap=None for RGB
        plt.title(filename)

        # Save the figure
        output_path = os.path.join(folder_name, f"{filename}.png")
        plt.savefig(output_path)
        plt.close()

# Process and plot for 'only_resize_testr'
process_and_save_images(only_resize_testr, testr_circles, svm_classifier, "src/TestR_HoG")

# Process and plot for 'only_resize_tesv'
process_and_save_images(only_resize_testv, testv_circles, svm_classifier, "src/TestV_HoG")


def plot_offsets_and_segmentation(image_dict, circles, output_folder):
    for idx, (filename, image) in enumerate(image_dict.items()):
        #print(f"Processing image: {filename}")
        
        # Check if there are circles for the current image
        if filename not in circles:
            #print(f"No circles data for image: {filename}")
            continue

        image_circles = circles[filename]
        image_segmented = image.copy()

        for x, y, r in image_circles:
            # Draw circle offsets on original image
            cv2.circle(image, (x, y), r, (0, 255, 0), 2)
            # Create segmentation map
            cv2.circle(image_segmented, (x, y), r, (255, 255, 255), -1)

        # Convert BGR to RGB for plotting if needed
        if len(image.shape) == 3 and image.shape[2] == 3:  # Check if the image is in color
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_segmented_rgb = cv2.cvtColor(image_segmented, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image
            image_segmented_rgb = image_segmented

        # Plot and save the images
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.imshow(image_rgb)
        plt.title('Offsets - ' + filename)
        plt.subplot(1, 2, 2)
        plt.imshow(image_segmented_rgb)
        plt.title('Segmentation Map - ' + filename)
        plt.tight_layout()

        # Save the figure
        output_path = os.path.join(output_folder, f"output_{idx}.png")
        plt.savefig(output_path)
        plt.close()

# Example usage
output_folderR= "src/TestR_HoG"
output_folderV= "src/TestV_HoG"


# Assuming testr_resized_images and testr_circles are your datasets
plot_offsets_and_segmentation(testr_resized_images, testr_circles, output_folderR)
plot_offsets_and_segmentation(testv_resized_images, testv_circles, output_folderV)
