import multiprocessing
from datetime import datetime
import numpy as np
import cv2
import subprocess

# Function that restart the computer to handle problem with USB cards getting full
def restart_system(s):
    import time
    print('Starting windows restart timer at: {}'.format(datetime.now().strftime("%Y_%m_%d#%H-%M-%S")))
    time.sleep(s)

    import os
    print('Windows restart!')
    os.system('shutdown -t 0 -r -f')

# Function that calls relay_run_feeder.exe which is a executable that tell the USB-relay to open and then close the ports for the feeder
def relay_run_feeder():
    subprocess.call(['Path to: relay_run_feeder.exe'])

# Function that calibrate the exposure in order to get good footage
def calibrate_exposure(target_pixel_sum, tolerance, current_exposure):

    exposure = current_exposure

    # Setting up camera
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
    cap.set(cv2.CAP_PROP_EXPOSURE, exposure)


    stop_frame = 30 # Frame n to stop at
    frame_n = 0 # Frame count

    # Camera loop that stop at stop_frame, used to collect a pixel sum after camera has stabilized
    while frame_n < stop_frame:
            ret, frame = cap.read()
            if ret:
                pixel_sum = np.sum(frame)
            frame_n += 1

    cap.release()

    # Check if the pixel sum is high or low, in other words decide whether to lower or higher the exposure
    try:
        if pixel_sum < target_pixel_sum - tolerance * target_pixel_sum and exposure < 15:
            exposure += 1
        elif pixel_sum > target_pixel_sum + tolerance * target_pixel_sum:
            exposure -= 1

        return exposure

    except TypeError:
        return exposure

# Function that write video
def write_video(filename, frames, fps):
    shape_frame = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    writer = cv2.VideoWriter(filename, fourcc, fps, (shape_frame[1], shape_frame[0]), isColor=True)

    for frame in frames:
        writer.write(frame)

    writer.release()

def detect_dropped_object(conn_bool, exposure):
    '''
    This function uses the camera inside the machine to detect if an items has been dropped. When the script is running
    this is a subprocess which sends a boolean to the other subprocess (the function responsible for the outdoor camera)
    when it is triggered. We are using a very simple movement detection, we compare the current frame with a background
    to detect changes. It takes the connection of a pipe to send a boolean through and exposure of the camera as input
    parameters.
    '''

    # Setting up the camera
    cap = cv2.VideoCapture(1)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
    cap.set(cv2.CAP_PROP_EXPOSURE, exposure)
    #cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    #cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    #cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    #cap.set(cv2.CAP_PROP_FOCUS, 10)

    print('Starting inside camera with exposure: ' + str(exposure))

    # Some parameters for the function
    frame_n = 0 # Frame count
    background_update = 10 # x frames then update
    start_frame = 30 * 6 # What frame_n to start, because the background has to stabilize and to fill the buffer. Has to be bigger or same as buffer_len
    background_th = 20 # Collect pixel values > x
    object_detection_th = 950000 # Threshold value for what is regarded an object falling down
    triggered = False # Boolean for multiprocess pipe

    # Camera loop
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        if ret:
            # Update background to current frame
            if frame_n % background_update == 0 or frame_n == start_frame:
                background = frame

            # After camera have stabilized, subtract background and sum the pixel intensity
            if frame_n > start_frame:
                img = cv2.subtract(frame, background)
                flat_img = img.flatten()
                img_filtered = flat_img[np.where(flat_img > background_th)]
                img_sum = float(np.sum(img_filtered))

                # Calculate difference in pixel sum from the last pixel sum to detect change (movement)
                if frame_n == start_frame + 1:
                    last_sum = img_sum
                diff = img_sum - last_sum
                last_sum = img_sum

                # Check if difference is big enough to be regarded as an object falling down
                if diff > object_detection_th:
                    triggered = True
                    conn_bool.send(triggered)
                    conn_bool.close()
                    print('Detection triggered!')
                    break

            frame_n += 1
            conn_bool.send(triggered)

        elif not ret:
            break

    # Release camera capture
    cap.release()

def capture_footage(conn_frames, conn_bool, exposure):
    '''
    This function uses the camera outside of the machine to capture the footage of what triggered the function that
    detects dropped objects. In order to get footage before and after the detection of dropped object we use a
    'frame buffer'. It takes 2 pipe connections, one to receive boolean from the 'detect dropped object'
     function and one to send the frames through as well as the exposure as input parametres.
    '''

    # Setting up the camera
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
    cap.set(cv2.CAP_PROP_EXPOSURE, exposure)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    cap.set(cv2.CAP_PROP_FOCUS, 10)

    print('Starting outside camera with exposure: ' + str(exposure))

    frame_n = 0 # Frame count
    start_frame = 30 * 6 # What frame_n to start, because the background has to stabilize and to fill the buffer. Has to be bigger or same as buffer_len
    frame_n_detection = 0 # Frame count since detection
    recording_len = 30 * 10 # How many frames should be captured after detection triggered, if fps = 30 then 300 frames = 10 seconds
    buffer_len = start_frame # How many frames should be captured before detection triggered
    buffer_split = None # Used to know where to split buffer to arrange the frames in proper order for footage
    pipe_bool = False # Boolean from pipe from 'detect dropped object' function
    error_bool = False # Boolean to check if error handling triggered
    triggered = False # Boolean to stop filling buffer when object detection is triggered

    # Arrays to store frames
    frames_recording = np.zeros((recording_len, 480, 640, 3), dtype=np.uint8)
    frames_buffer = np.zeros((buffer_len, 480, 640, 3), dtype=np.uint8)

    # Camera loop
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        if ret:
            # Filling up the buffer
            if not triggered:
                frames_buffer[frame_n % buffer_len] = frame

            # If 'detect dropped object' function is triggered stop filling buffer and start recording the aftermath
            if not triggered and pipe_bool:
                triggered = True
                buffer_split = frame_n % buffer_len
                feeder.start() # Start subprocess which run feeder
                print("Triggered at: ", datetime.now().strftime("%Y_%m_%d#%H-%M-%S"))
                frame_n_detection = 0

            # Getting frames for aftermath recording
            if triggered:
                # Saving frames for writing recording
                frames_recording[frame_n_detection] = frame
                frame_n_detection += 1

            # Stop recording if all frames are collected
            if recording_len == frame_n_detection and triggered:
                break

            frame_n += 1

            # Check if 'detect dropped object' function is triggered
            if not pipe_bool:
                pipe_bool = conn_bool.recv()

        elif not ret:
            error_bool = True
            break

    # Release camera capture
    cap.release()

    if error_bool:
        print('Error handling triggered!')

    if not error_bool:
        print('Stopped recording!')

        feeder.join() # Wait and make sure the feeder has stopped

        # Arrange buffer and concatenate all frames
        fixed_buffer = np.concatenate((frames_buffer[buffer_split + 1:, :, :, :], frames_buffer[:buffer_split + 1, :, :, :]), axis=0)
        all_frames = np.concatenate((fixed_buffer, frames_recording), axis=0)

        conn_frames.send(all_frames)
        conn_frames.close()


# Creating multiprocess for feeder and creating pipes
feeder = multiprocessing.Process(target=relay_run_feeder)

parent_conn_frames, child_conn_frames = multiprocessing.Pipe()
parent_conn_bool, child_conn_bool = multiprocessing.Pipe()


t_restart = 60 * 60 * 8 # x hours between restart
restart_timer = multiprocessing.Process(target=restart_system, args=(t_restart,)) # Creating multiprocess

current_exposure = -12 # Exposure parameter

if __name__ == '__main__':

    restart_timer.start()

    # System loop
    while True:

        temp_exposure = calibrate_exposure(9*10**7, 0.3, current_exposure)

        indCam = multiprocessing.Process(target=detect_dropped_object, args=(child_conn_bool, -9,))
        outCam = multiprocessing.Process(target=capture_footage, args=(child_conn_frames, parent_conn_bool, temp_exposure,))

        indCam.start()
        outCam.start()

        try:
            video_frames = parent_conn_frames.recv()
            write_video('path to video folder/{}.avi'.format(datetime.now().strftime("%Y_%m_%d#%H-%M-%S")), video_frames , 30)

            indCam.join()
            outCam.join()

            print('Video saved.')
            current_exposure = temp_exposure

        except:
            pass
