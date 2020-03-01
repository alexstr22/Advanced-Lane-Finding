# Self-Driving Car Engineer Nanodegree
---
## Advanced lane finding
![Lanes Image](./examples/example_output.jpg)
---
### Steps
#### My pipeline consisted od 10 steps:

1) Import packages
2) Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
3) Apply a distortion correction to raw images.
4) Use color transforms, gradients, etc., to create a thresholded binary image.
5) Apply a perspective transform to rectify binary image ("birds-eye view").
6) Detect lane pixels and fit to find the lane boundary.
7) Determine the curvature of the lane and vehicle position with respect to center.
8) Warp the detected lane boundaries back onto the original image.
9) Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.
10) Run pipeline on video


The code for this step is contained in the first code cell of the IPython notebook located in "./examples/example.ipynb" 

[//]: # (Image References)

[image1]: ./report_img/Undistorted_image.png "Undistorted"
[image2]: ./report_img/Thresholded_gradient_orient_x.png "gradient_orient_x"
[image3]: ./report_img/Thresholded_gradient_orient_y.png "gradient_orient_y"
[image4]: ./report_img/Thresholded_magnitude.png "Thresholded_magnitude"
[image5]: ./report_img/Thresholded_gradient_direction.png  "Thresholded_gradient_direction"
[image6]: ./report_img/Color_thresholded.png "Color_thresholded"
[image7]: ./report_img/Thresholds_combined.png "Thresholds_combined"
[image8]: ./report_img/birds-eye.png "birds-eye"
[image9]: ./report_img/Histogram.png "Histogram"
[image10]: ./report_img/Lane_lines_detected.png "Lane_lines_detected"
[image11]: ./report_img/Lane_detected.png "Lane_detected"
[image12]: ./report_img/Lane_detected_with_metrics.png "Lane_detected_with_metrics"

[video1]: './project_video_solution.mp4'

---


## Pipeline (image)
---
### Import packages

OpenCV - an open source computer vision library,
Matplotbib - a python 2D plotting libray,
Numpy - a package for scientific computing with Python,
MoviePy - a Python module for video editing.


### Step 1: Camera Calibration


The next step is to calibrate the camera. A set of chess images will be used for this.

I have defined the `calibrate_camera` function which takes as input parameters an array of paths to chessboards images, and the number of inside corners in the _x_ and _y_ axis.

For each image path, `calibrate_camera`:
 * reads the image by using the OpenCV function - cv2.read(), 
 * converts it to grayscale usign - cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) , 
 * find the chessboard corners usign - cv2.findChessboardCorners(gray, (nx, ny), None)
 * calibrate the camera  on all images by function - cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)


The values returned by `cv2.calibrateCamera` will be used later to undistort our video images.

The complete code for this step can be found in the Advanced_Lane_Finding.ipynb cell with name 'calibrate_camera'


### Step 2: Distortion correction

In this step we use undistor function from opencv - cv2.undistort(img, mtx, dist, None, mtx)

Below you can see a result on chessboard images
![alt text][image1]

### Step 3: Use color transforms, gradients, etc., to create a thresholded binary image.
#### Step 3.1: Directional gradient

In function 'abs_sobel_thresh' we use sobel filter by 'np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0))' to colculate directional gradient.
Function takes as input parameters x or y as parameter for orien gradient

* Thresholded gradient orient x
![alt text][image2]
* Thresholded gradient orient y
![alt text][image3]

#### Step 3.2  Magnitude of the Gradient

Define a function that applies Sobel x and y (cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel) and cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel) )
then computes the magnitude 'np.sqrt(sobelx^2 + sobely^2)' of the gradient and applies a threshold


![alt text][image4]


#### Step 3.3: Gradient direction

Calculate the x and y gradients by
`<
sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
>`

Take the absolute value of the gradient direction by  np.arctan2(np.absolute(sobely), np.absolute(sobelx)),
apply a threshold, and create a binary image result

![alt text][image5]

#### Step 3.4: Color threshold

Apply function cv2.cvtColor(img, cv2.COLOR_RGB 2 HLS) to convert RGB on HLS and simple choose threshold, then return output.

![alt text][image6]


#### Step 3.5: Combine all the thresholds to identify the lane lines

Function 'combine thresholds' is combined previous steps.

The result of this function you can see below

![alt text][image7]


#### Step 4: Perspective transform ("birds-eye view")

The next step in our pipeline is to transform our sample image to _birds-eye_ view.

In this case i extracted the vertices to perform a perspective transformation. Destinations are chosen so that the straight lines look more or less parallel on the transformed image. The Opencv function cv2.getPerspectiveTransform will be used to calculate both the corresponding M transformation and the inverse Minv transformation. M and Minv will be used respectively for warping and expanding video images.

Please find below the result of warping an image after transforming its perpective to birds-eye view:

![alt text][image8]


#### Step 5: Identified lane-line pixels and fit their positions with a polynomial
The first step: plot the histogram as a starting point for determining where the lane lines are. (Find the peak of the left and right halves of the histogram). Than we set up windows for shifting. Then use sliding windows moving upward in the imag to determine where the lane lines go. Given window sliding left or right if it finds the mean position of activated pixels within the window to have shifted.
(If you found > minpix pixels, recenter next window on their mean position). This technique known as Sliding Window is used to identify the most likely coordinates of the lane lines in a window, which slides vertically through the image for both the left and right line.
(see find_lane_pixels function)
Finally, usign the coordinates previously calculated, a second order polynomial is calculated for both the left and right lane line. 
Fit a second order polynomial to each using `np.polyfit`

`left_fit = np.polyfit(lefty, leftx, 2)
right_fit = np.polyfit(righty, rightx, 2)
ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
left_fitx = left_fit[0]*ploty^2 + left_fit[1]*ploty + left_fit[2]
right_fitx = right_fit[0]*ploty^2 + right_fit[1]*ploty + right_fit[2]`

(see fit_polynomial function )

![alt text][image9]


Once you have selected the lines, it is reasonable to assume that the lines will remain there in future video frames. search_around_poly() uses the previosly calculated line_fits to try to identify the lane lines in a consecutive image.The green shaded area shows where we searched for the lines this time.

![alt text][image10]



#### Step 6: Determine the curvature of the lane and vehicle position with respect to center


 Now let's calculate the radius of curvature and the car offset.
The radius of curvature is computed according to the formula and method described in the classroom material. 

`def curvature_radius (leftx, rightx, img_shape, xm_per_pix=3.7/800, ym_per_pix = 25/720):
    ploty = np.linspace(0, img_shape[0] - 1, img_shape[0])
    
    leftx = leftx[::-1]  # Reverse to match top-to-bottom in y
    rightx = rightx[::-1]  # Reverse to match top-to-bottom in y
    
    # Fit a second order polynomial to pixel positions in each fake lane line
    left_fit = np.polyfit(ploty, leftx, 2)
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fit = np.polyfit(ploty, rightx, 2)
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 25/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/800 # meters per pixel in x dimension

    # Fit new polynomials to x,y in world space
    y_eval = np.max(ploty)
    left_fit_cr = np.polyfit(ploty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty*ym_per_pix, rightx*xm_per_pix, 2)
    
    # Calculate the new radii of curvature
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    
    # Now our radius of curvature is in meters
    return (left_curverad, right_curverad)`





#### Step 7:Warp the detected lane boundaries back onto the original image

We have already identified the lane lines, its radius of curvature and the car offset.
And now we can draw the lanes on the original image. (use function draw_lane() with OpenCV)

![alt text][image11]

#### Step 8: Display lane boundaries and numerical estimation of lane curvature and vehicle position
The next step is to add metrics to the image.
Use openCV function cv2.putText.

![alt text][image12]


---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [video1](./project_video_solution.mp4)

---

### Discussion

Ideas for improvement
-More accurate selection of filter hyperparameters
-I think can work on improving the function - "search_around_poly"
-Apply neural networks ( I looked into the next lesson :) )
