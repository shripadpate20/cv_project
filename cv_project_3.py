#!/usr/bin/env python
# coding: utf-8

# In[2]:


import cv2
import numpy as np


# In[3]:


def init_video(input_path):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise IOError(f"Could not open video file '{input_path}'")
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    return cap, fps, frame_width, frame_height


# In[4]:


def create_video_writer(output_path, fps, frame_width, frame_height):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    return cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))


# In[5]:


def init_bg_subtractor():
    return cv2.createBackgroundSubtractorMOG2()


# In[6]:


def compute_optical_flow(prev_frame, curr_frame):
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    return flow


# In[7]:


def compute_motion_mask(flow, threshold=2.0):
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    return mag > threshold


# In[8]:


def detect_and_draw_motion(curr_frame, flow, motion_mask, bg_subtractor):
    fg_mask = bg_subtractor.apply(curr_frame)
    fg_mask = cv2.bitwise_and(fg_mask, fg_mask, mask=motion_mask.astype(np.uint8) * 255)
    
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    result_frame = curr_frame.copy()
    for contour in contours:
        if cv2.contourArea(contour) > 1000:  # Filter small contours
            # Compute bounding box and centroid
            x, y, w, h = cv2.boundingRect(contour)
            cx, cy = x + w // 2, y + h // 2
            
            # Compute average velocity within the bounding box
            avg_velocity = np.mean(flow[y:y + h, x:x + w], axis=(0, 1))
            velocity_magnitude = np.linalg.norm(avg_velocity)
            
            # Determine box color based on velocity magnitude
            if velocity_magnitude > 5.0:
                box_color = (0, 0, 255)  # Red (fast)
            elif velocity_magnitude < 2.0:
                box_color = (0, 255, 0)  # Green (slow)
            else:
                box_color = (0, 255, 255)  # Yellow (medium)
            
            # Draw bounding box and optical flow vector
            cv2.rectangle(result_frame, (x, y), (x + w, y + h), box_color, 2)
            flow_start = (cx, cy)
            flow_end = (int(cx + avg_velocity[0]), int(cy + avg_velocity[1]))
            cv2.arrowedLine(result_frame, flow_start, flow_end, (255, 255, 0), 2)
    
    return result_frame


# In[9]:


def draw_optical_flow_field(frame, flow, step=16):
    h, w = frame.shape[:2]
    y, x = np.mgrid[step//2:h:step, step//2:w:step].reshape(2, -1).astype(int)
    fx, fy = flow[y, x].T
    lines = np.vstack([x, y, x + fx, y + fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    cv2.polylines(frame, lines, 0, (0, 255, 0), thickness=1, lineType=cv2.LINE_AA)


# In[10]:


def process_video(input_path, output_path):
    try:
        cap, fps, frame_width, frame_height = init_video(input_path)
        out = create_video_writer(output_path, fps, frame_width, frame_height)
        bg_subtractor = init_bg_subtractor()
        
        ret, prev_frame = cap.read()
        if not ret:
            raise IOError("Failed to read the first frame")

        while True:
            ret, curr_frame = cap.read()
            if not ret:
                break
            
            flow = compute_optical_flow(prev_frame, curr_frame)
            motion_mask = compute_motion_mask(flow)
            
            result_frame = detect_and_draw_motion(curr_frame, flow, motion_mask, bg_subtractor)
            draw_optical_flow_field(result_frame, flow, step=16)
            
            out.write(result_frame)
            
            prev_frame = curr_frame
        
        cap.release()
        out.release()
        print(f"Processed video saved: {output_path}")

    except Exception as e:
        print(f"Error during video processing: {str(e)}")


# In[12]:


if __name__ == "__main__":
    input_video_path = 'sun17.mp4'
    output_video_path = 'cv_project_3_1.mp4'
    process_video(input_video_path, output_video_path)


# In[ ]:




