########### Python 2.7 #############
import cv2
import os

    
def create_video(images_path, output_file = "out.avi", fps = 5):
    # Determine the width and height from the first image
    if len(images_path) == 0:
        return
    frame = cv2.imread(images_path[0])
#    print (images_path[0])
#    print (frame)
#    print frame
#    cv2.imshow('video',frame)
    height, width, channels = frame.shape
    
    # Define the codec and create VideoWriter object
#    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Be sure to use lower case
    #out = cv2.VideoWriter(output, fourcc, 20.0, (width, height))
    #out = cv2.VideoWriter('output.avi',cv2.VideoWriter_fourcc(*'XVID'), 20.0, (640,480))
    out = cv2.VideoWriter(output_file,
                          cv2.VideoWriter_fourcc('M','J','P','G'), fps, (width,height))
    for im_path in images_path:
        frame = cv2.imread(im_path)
    #    print frame.shape
        out.write(frame) # Write out frame to video
    
#        cv2.imshow('video',frame)
#        if (cv2.waitKey(1) & 0xFF) == ord('q'): # Hit `q` to exit
#            break
    
    # Release everything if job is finished
    out.release()
#    cv2.destroyAllWindows()
