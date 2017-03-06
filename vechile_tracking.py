import os, time, glob
import numpy as np
import matplotlib.pyplot as plt

from moviepy.video.io.VideoFileClip import VideoFileClip
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from scipy.ndimage.measurements import label
from lesson_functions import *

#
#
windows = []
svc = None
X_scaler = None

#
# TODO: Tweak these parameters and see how the results change.
color_space = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 8  # HOG orientations
pix_per_cell = 8 # HOG pixels per cell
cell_per_block = 2 # HOG cells per block
hog_channel = 'ALL' # Can be 0, 1, 2, or "ALL"
spatial_size = (32, 32) # Spatial binning dimensions
hist_bins = 16    # Number of histogram bins
spatial_feat = True # Spatial features on or off
hist_feat = True # Histogram features on or off
hog_feat = True # HOG features on or off

#
#
def load_image_and_extract_features(plot = 0):
    cars = []
    notcars = []
    images = []

    for root, dirs, files in os.walk('data'):
        for file in files:
            if file.endswith(".png"):
                images.append(os.path.join(root, file))

    for img in images:
        if 'non-vehicles' in img:
            notcars.append(img)
        else:
            cars.append(img)

    car_features = extract_features(cars, color_space=color_space,
                                    spatial_size=spatial_size, hist_bins=hist_bins,
                                    orient=orient, pix_per_cell=pix_per_cell,
                                    cell_per_block=cell_per_block,
                                    hog_channel=hog_channel, spatial_feat=spatial_feat,
                                    hist_feat=hist_feat, hog_feat=hog_feat)

    notcar_features = extract_features(notcars, color_space=color_space,
                                        spatial_size=spatial_size, hist_bins=hist_bins,
                                        orient=orient, pix_per_cell=pix_per_cell,
                                        cell_per_block=cell_per_block,
                                        hog_channel=hog_channel, spatial_feat=spatial_feat,
                                        hist_feat=hist_feat, hog_feat=hog_feat)


    if plot == 1:
        img = mpimg.imread(cars[0])
        hog_features, hog_image = get_hog_features(img[:, :, 1], orient,
                                        pix_per_cell, cell_per_block, vis=True, feature_vec=True)
        plt.imshow(img)
        plt.show()
        plt.imshow(hog_image)
        plt.show()

        img = mpimg.imread(notcars[0])
        hog_features, hog_image = get_hog_features(img[:, :, 1], orient,
                                                   pix_per_cell, cell_per_block, vis=True, feature_vec=True)
        plt.imshow(img)
        plt.show()
        plt.imshow(hog_image)
        plt.show()

    return car_features, notcar_features

#
#
def classify(car_features, notcar_features):
    global X_scaler
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))
    X = np.vstack((car_features, notcar_features)).astype(np.float64)
    # Fit a per-column scaler
    X_scaler = StandardScaler().fit(X)
    # Apply the scaler to X
    scaled_X = X_scaler.transform(X)

    rand_state = np.random.randint(0, 100)
    x_train, x_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.2, random_state=rand_state)

    # Use a linear SVC
    svc = LinearSVC()
    # Check the training time for the SVC
    t = time.time()
    svc.fit(x_train, y_train)
    t2 = time.time()
    print(round(t2 - t, 2), 'Seconds to train SVC...')
    # Check the score of the SVC
    print('Test Accuracy of SVC = ', round(svc.score(x_test, y_test), 4))

    return svc


#
#
def process_image(img, plot = 0):
    global windows, svc, X_scaler
    draw_img = np.copy(img)
    img = img.astype(np.float32) / 255

    heat = np.zeros_like(img[:, :, 0]).astype(np.float)

    hot_windows = search_windows(img, windows, svc, X_scaler, color_space=color_space,
                                 spatial_size=spatial_size, hist_bins=hist_bins,
                                 orient=orient, pix_per_cell=pix_per_cell,
                                 cell_per_block=cell_per_block,
                                 hog_channel=hog_channel, spatial_feat=spatial_feat,
                                 hist_feat=hist_feat, hog_feat=hog_feat)

    heat = add_heat(heat, hot_windows)
    heat = apply_threshold(heat, 2)

    if plot == 1:
        plt.imshow(heat)
        plt.show()

    labels = label(heat)

    if plot == 1:
        plt.imshow(labels[0], cmap='gray')
        plt.show()

    return draw_labeled_bboxes(draw_img, labels)

#
#
def process_video(video):
    in_video  = "{}.mp4".format(video)
    out_video = "{}_output.mp4".format(video)

    clip = VideoFileClip(in_video)
    output_clip = clip.fl_image(process_image)
    output_clip.write_videofile(out_video, audio=False)

#
#
def main():
    global svc, windows
    window_sizes = [
        #32,
        64,
        96,
        128
    ]

    y_start_stop = [ [392, 504],
                     #[392, 576],
                     [392, 576],
                     [392, 720] ]

    car_features, notcar_features = load_image_and_extract_features(plot = 0)
    svc = classify(car_features, notcar_features)

    print("Classify Done")

    #print(img.shape)
    img_shape = [720, 1280]

    for i in range(len(window_sizes)):
        windows += slide_window(
            img_shape, x_start_stop=[320, None], y_start_stop=y_start_stop[i],
            xy_window=(window_sizes[i], window_sizes[i]), xy_overlap=(0.8, 0.8)
        )

    #print("Staring Image Process")
    #img = mpimg.imread("test_images/test4.jpg")
    #window_img = draw_boxes(img, windows, color=(0, 0, 255), thick=6)
    #plt.imshow(window_img)
    #plt.show()
    #draw_img = process_image(img, plot = 1)
    #print("Done Process Image")
    #plt.imshow(draw_img)
    #plt.show()
    process_video("project_video")


if __name__ == "__main__":
    main()