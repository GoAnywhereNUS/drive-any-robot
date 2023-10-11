import rospy
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, Float32MultiArray

import torch
import cv2
from PIL import Image as PILImage
import numpy as np
import os
import argparse
import yaml
import time
import bisect
import itertools
from collections import deque

from utils import msg_to_pil, to_numpy, transform_images, load_model

MODEL_WEIGHTS_PATH = "../model_weights"
ROBOT_CONFIG_PATH ="../config/spot.yaml"
MODEL_CONFIG_PATH = "../config/models.yaml"
with open(ROBOT_CONFIG_PATH, "r") as f:
    robot_config = yaml.safe_load(f)
MAX_V = robot_config["max_v"]
MAX_W = robot_config["max_w"]
RATE = robot_config["frame_rate"] 
IMAGE_TOPIC = "/rs_mid/color/image_raw"
ODOM_TOPIC = "/spot/odometry"

# DEFAULT MODEL PARAMETERS (can be overwritten by model.yaml)
model_params = {
    "path": "gnm_large.pth", # path of the model in ../model_weights
    "model_type": "gnm", # gnm (conditioned), stacked, or siamese
    "context": 5, # number of images to use as context
    "len_traj_pred": 5, # number of waypoints to predict
    "normalize": True, # bool to determine whether or not normalize images
    "image_size": [85, 64], # (width, height)
    "normalize": True, # bool to determine whether or not normalize the waypoints
    "learn_angle": True, # bool to determine whether or not to learn/predict heading of the robot
    "obs_encoding_size": 1024, # size of the encoding of the observation [only used by gnm and siamese]
    "goal_encoding_size": 1024, # size of the encoding of the goal [only used by gnm and siamese]
    "obsgoal_encoding_size": 2048, # size of the encoding of the observation and goal [only used by stacked model]
    "min_linear_vel": 0.05, # minimum linear velocity above which the robot is deemed to be moving
    "min_angular_vel": 0.03, # minimum angular velocity above which the robot is deemed to be moving
}

# GLOBALS
odom_queue = deque(maxlen=30) # Buffer of past odometry messages (assuming 30Hz sim)
history_queue = deque(maxlen=max(30, model_params["context"] + 1)) # Buffer of received sensor images
context_queue = None # Buffer of selected sensor images for embodiment context
zupt_queue = deque(maxlen=max(30, model_params["context"] + 1)) # Buffer of flags tagged to each element of history queue, indicating if robot has zero velocity
tlx, tly = -1, -1
brx, bry = -1, -1
movex, movey = -1, -1
image_crop = None
drawing = False

# Load the model (locobot uses a NUC, so we can't use a GPU)
device = torch.device("cuda")

def find_closest_odom(query_timestamp):
    if len(odom_queue) == 0:
        return None
    
    timestamps = [ts for ts, _ in odom_queue]

    # Use bisect_left to find the index where the query_timestamp would be inserted
    index = bisect.bisect_left(timestamps, query_timestamp)

    if index == 0:
        return odom_queue[0][1] # query_timestamp is before the first datum in the buffer
    elif index == len(odom_queue):
        return odom_queue[-1][1] # query_timestamp is after the last datum in the buffer
    else:
        # query_timestamp falls between two timestamps in the buffer
        # Determine which datum is closer to the query_timestamp
        left_timestamp = odom_queue[index - 1][0]
        right_timestamp = odom_queue[index][0]

        if abs(query_timestamp - left_timestamp) <= abs(query_timestamp - right_timestamp):
            return odom_queue[index - 1][1] # Return left datum
        else:
            return odom_queue[index][1] # Return right datum
        
def find_recent_sequence():
    seq_end = None
    seq_count = 0

    # Iterate over the queues in reverse
    for i in range(len(history_queue) - 1, -1, -1):
        if zupt_queue[i]:
            seq_count = 0
            seq_end = None
        elif seq_end is None:
            # First element of a sequence with zupt False flag
            seq_count += 1
            seq_end = i
        else:
            # Element in a sequence with zupt False flag
            seq_count += 1
            if seq_count == model_params["context"] + 1:
                return list(itertools.islice(history_queue, i, seq_end + 1))

    return None
    
def image_callback(msg):
    nearest_odom = find_closest_odom(msg.header.stamp)
    if nearest_odom is None:
        return

    if (abs(nearest_odom.twist.twist.linear.x) < model_params["min_linear_vel"]
        and abs(nearest_odom.twist.twist.angular.z) < model_params["min_angular_vel"]):
        zupt_queue.append(True)
    else:
        zupt_queue.append(False)

    obs_img = msg_to_pil(msg)
    history_queue.append(obs_img)

# def image_callback(msg):
#     obs_img = msg_to_pil(msg)
#     if len(history_queue) < model_params["context"] + 1:
#         history_queue.append(obs_img)
#     else:
#         history_queue.pop(0)
#         history_queue.append(obs_img)

def odom_callback(msg):
    odom_queue.append((msg.header.stamp, msg))

# Mouse callback function to select rectangular region
def draw_rectangle(event, x, y, flags, param):
    global tlx, tly, brx, bry, movex, movey, history_queue, image_crop, drawing
    if event == cv2.EVENT_MOUSEMOVE:
        movex, movey = x, y
    elif event == cv2.EVENT_LBUTTONDOWN:
        tlx, tly = x, y
        movex, movey = x, y
        drawing = True
    elif event == cv2.EVENT_LBUTTONUP:
        if tlx == x or tly == y:
            return

        if tlx < x: tlx, brx = tlx, x
        else: tlx, brx = x, tlx
        if tly < y: tly, bry = tly, y
        else: tly, bry = y, tly

        if len(history_queue) > 1:
            img = np.asarray(history_queue[-1])
            cv2.namedWindow("Selected")
            cv2.imshow("Selected", img[tly:bry, tlx:brx])
            
            cv2.namedWindow("Crop")
            cropped = bbox2crop(img, tlx, tly, brx, bry)
            cv2.imshow("Crop", cropped)

            image_crop = cropped

        tlx, tly = -1,-1
        brx, bry = -1,-1
        movex, movey = -1,-1
        drawing = False

# Fit a given bounding box inside an image crop with
# same aspect ratio as original image
def bbox2crop(img, tlx, tly, brx, bry):
    cx = 0.5 * (tlx + brx)
    cy = 0.5 * (tly + bry)
    ih, iw, _ = img.shape
    bboxw, bboxh = brx - tlx, bry - tly
    img_ar = float(iw) / float(ih)
    bbox_ar = float(bboxw) / float(bboxh)

    if img_ar > bbox_ar:
        # Fit bbox's height within the crop
        cropw, croph = img_ar * bboxh, bboxh
    else:
        # Fit bbox's width within the crop
        cropw, croph = bboxw, bboxw / img_ar

    tlx = int(max(0, cx - (0.5 * cropw)))
    brx = int(min(iw-1, cx + (0.5 * cropw)))
    tly = int(max(0, cy - (0.5 * croph)))
    bry = int(min(ih-1, cy + (0.5 * croph)))

    return img[tly:bry, tlx:brx]


def main(args: argparse.Namespace):
    # load model parameters
    with open(MODEL_CONFIG_PATH, "r") as f:
        model_config = yaml.safe_load(f)
    for param in model_config:
        model_params[param] = model_config[param]

    # load model weights
    model_filename = model_config[args.model]["path"]
    model_path = os.path.join(MODEL_WEIGHTS_PATH, model_filename)
    if os.path.exists(model_path):
        print(f"Loading model from {model_path}")
    else:
        raise FileNotFoundError(f"Model weights not found at {model_path}")
    model = load_model(
        model_path,
        model_params["model_type"],
        model_params["context"],
        model_params["len_traj_pred"],
        model_params["learn_angle"], 
        model_params["obs_encoding_size"], 
        model_params["goal_encoding_size"],
        model_params["obsgoal_encoding_size"],
        device
    )
    model.eval()

    print("Loaded model of size: ", sum(p.numel() for p in model.parameters()))

    # Set up ROS node
    rospy.init_node("gnm", anonymous=False)
    rate = rospy.Rate(RATE)
    image_sub = rospy.Subscriber(IMAGE_TOPIC, Image, image_callback, queue_size=1)
    odom_sub = rospy.Subscriber(ODOM_TOPIC, Odometry, odom_callback, queue_size=5)
    waypoint_pub = rospy.Publisher("/gnm/waypoint", Float32MultiArray, queue_size=5)
    goal_pub = rospy.Publisher("/gnm/reached_goal", Bool, queue_size=5)

    global image_crop, context_queue
    cv2.namedWindow("Viz")
    cv2.setMouseCallback("Viz", draw_rectangle)

    while not rospy.is_shutdown():
        if len(history_queue) > 1:
            # Selection and visualisation of image cropping
            curr_im = np.asarray(history_queue[-1])
            if drawing and (movex, movey) != (-1, -1):
                cv2.rectangle(curr_im, (tlx, tly), (movex, movey), (0, 255, 0), 2)
            cv2.imshow("Viz", curr_im)
            if cv2.waitKey(1) == ord('q'):
                break
            if cv2.waitKey(1) == ord('s'):
                image_crop = None

            # Update context_queue with valid embodiment context (i.e. sequence of non-zero velocities)
            if args.only_moving_contexts:
                nearest_sequence = find_recent_sequence()
                if nearest_sequence is not None:
                    rospy.loginfo("Context queue updated!")
                    context_queue = nearest_sequence
            else:
                context_queue = (None if len(history_queue) < model_params["context"] + 1 
                                 else list(itertools.islice(history_queue, -(model_params["context"] + 1), len(history_queue))))
            
            # If there is an image crop and valid embodiment context, run GNM
            if image_crop is not None and context_queue is not None:
                    rospy.loginfo("Received crop!")
                    pil_crop = PILImage.fromarray(np.uint8(image_crop))
                    transf_goal_img = transform_images(pil_crop, model_params["image_size"]).to(device)
                    transf_obs_img = transform_images(context_queue, model_params["image_size"]).to(device)
                    start = time.time()
                    dist, waypoint = model(transf_obs_img, transf_goal_img) 
                    dist = to_numpy(dist[0])
                    waypoint = to_numpy(waypoint[0])
                    
                    #print(">>> Took: ", time.time() - start)
                    #print("Dists:")
                    #print(dist)
                    #print("Waypoints:")
                    #print(waypoint)

                    waypoint_msg = Float32MultiArray()
                    chosen_waypoint = waypoint[args.waypoint]
                    if model_params["normalize"]:
                        chosen_waypoint[:2] *= (MAX_V / RATE)
                    waypoint_msg.data = chosen_waypoint
                    waypoint_pub.publish(waypoint_msg)
                    rospy.loginfo("Published waypoint: ")

        rate.sleep()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        "-m",
        default="gnm_large",
        type=str,
        help="model name (hint: check ../config/models.yaml) (default: large_gnm)",
    )
    parser.add_argument(
        "--waypoint",
        "-w",
        default=2, # close waypoints exihibit straight line motion (the middle waypoint is a good default)
        type=int,
        help=f"""index of the waypoint used for navigation (between 0 and 4 or 
        how many waypoints your model predicts) (default: 2)""",
    )
    parser.add_argument(
        "--only_moving_contexts",
        default=True,
        type=bool,
        help=f"""If true, the context will be the last contiguous sequence of
        images where the robot's velocity is non-zero. Otherwise it is the most
        recent sequence of images."""
    )
    args = parser.parse_args()
    main(args)
