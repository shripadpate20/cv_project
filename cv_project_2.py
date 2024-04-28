#!/usr/bin/env python
# coding: utf-8

# In[30]:


import cv2
import numpy as np


# In[ ]:


def initialize_video_writer(input_path, output_path):
    
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Error: Could not open video file '{input_path}'")
    
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    
    return cap, out


# In[ ]:


def read_first_frame(cap):
    
    ret, frame = cap.read()
    if not ret:
        raise ValueError("Error: Failed to read the first frame.")
    return frame


# In[ ]:


def calculate_optical_flow(prev_gray, curr_gray):
   
    return cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)


# In[ ]:


def process_frame(curr_frame, flow, bg_subtractor, mag_threshold=2.0, contour_area_threshold=1000):
    
    
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    
   
    motion_mask = (mag > mag_threshold).astype(np.uint8)
    
    
    fg_mask = bg_subtractor.apply(curr_frame)
    fg_mask = cv2.bitwise_and(fg_mask, fg_mask, mask=motion_mask * 255)
    
    
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    
    result_frame = curr_frame.copy()
    
    for contour in contours:
        if cv2.contourArea(contour) > contour_area_threshold:
            
            x, y, w, h = cv2.boundingRect(contour)
            
            
            cx, cy = x + w // 2, y + h // 2
            
            
            avg_velocity = np.mean(flow[y:y + h, x:x + w], axis=(0, 1))
            
           
            velocity_magnitude = np.linalg.norm(avg_velocity)
            
           
            if velocity_magnitude > 5.0:  # High speed (red)
                box_color = (0, 0, 255)  # Red
            elif velocity_magnitude < 2.0:  # Low speed (green)
                box_color = (0, 255, 0)  # Green
            else:  # Medium speed (yellow)
                box_color = (0, 255, 255)  # Yellow
            
            
            cv2.rectangle(result_frame, (x, y), (x + w, y + h), box_color, 2)
            
           
            flow_start = (int(cx), int(cy))
            flow_end = (int(cx + avg_velocity[0]), int(cy + avg_velocity[1]))
            cv2.arrowedLine(result_frame, flow_start, flow_end, (255, 255, 0), 2)

    return result_frame


# In[ ]:


def process_video(input_path, output_path):
    
    cap, out = initialize_video_writer(input_path, output_path)
    bg_subtractor = cv2.createBackgroundSubtractorMOG2()

    
    prev_frame = read_first_frame(cap)

    
    while True:
        ret, curr_frame = cap.read()
        if not ret:
            break

        
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

        
        flow = calculate_optical_flow(prev_gray, curr_gray)

       
        result_frame = process_frame(curr_frame, flow, bg_subtractor)

        
        out.write(result_frame)

        
        prev_frame = curr_frame

    
    cap.release()
    out.release()

    print(f"Processed video saved: {output_path}")


# In[ ]:


def draw_optical_flow_field(frame, flow, step=16):
    """Draw optical flow field on the frame for visualization."""
    h, w = frame.shape[:2]
    y, x = np.mgrid[step//2:h:step, step//2:w:step].reshape(2, -1).astype(int)
    fx, fy = flow[y, x].T
    lines = np.vstack([x, y, x + fx, y + fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    cv2.polylines(frame, lines, 0, (0, 255, 0), thickness=1, lineType=cv2.LINE_AA)


if __name__ == "__main__":
    input_video_path = 'sun17.mp4'
    output_video_path = 'cv_project_2_1.mp4'

    process_video(input_video_path, output_video_path)


# In[ ]:




