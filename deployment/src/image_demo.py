import rospy
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
}

# GLOBALS
context_queue = []
tlx, tly = -1, -1
brx, bry = -1, -1
movex, movey = -1, -1
image_crop = None
drawing = False

# # Load the model (locobot uses a NUC, so we can't use a GPU)
# device = torch.device("cpu")
device = torch.device("cuda")

def image_callback(msg):
    obs_img = msg_to_pil(msg)
    if len(context_queue) < model_params["context"] + 1:
        context_queue.append(obs_img)
    else:
        context_queue.pop(0)
        context_queue.append(obs_img)

# Mouse callback function to select rectangular region
def draw_rectangle(event, x, y, flags, param):
    global tlx, tly, brx, bry, movex, movey, context_queue, image_crop, drawing
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

        if len(context_queue) > 1:
            img = np.asarray(context_queue[-1])
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
    waypoint_pub = rospy.Publisher("/gnm/waypoint", Float32MultiArray, queue_size=1)
    goal_pub = rospy.Publisher("/gnm/reached_goal", Bool, queue_size=1)

    global image_crop
    cv2.namedWindow("Viz")
    cv2.setMouseCallback("Viz", draw_rectangle)

    while not rospy.is_shutdown():
        if len(context_queue) > 1:
            curr_im = np.asarray(context_queue[-1])
            if drawing and (movex, movey) != (-1, -1):
                cv2.rectangle(curr_im, (tlx, tly), (movex, movey), (0, 255, 0), 2)
            cv2.imshow("Viz", curr_im)
            if cv2.waitKey(1) == ord('q'):
                break
            if cv2.waitKey(1) == ord('s'):
                image_crop = None

            if image_crop is not None and len(context_queue) > model_params["context"]:
                    rospy.loginfo("Received crop!")
                    pil_crop = PILImage.fromarray(np.uint8(image_crop))
                    transf_goal_img = transform_images(pil_crop, model_params["image_size"]).to(device)
                    transf_obs_img = transform_images(context_queue, model_params["image_size"]).to(device)
                    start = time.time()
                    dist, waypoint = model(transf_obs_img, transf_goal_img) 
                    dist = to_numpy(dist[0])
                    waypoint = to_numpy(waypoint[0])
                    
                    #print(">>> Took: ", time.time() - start)
                    print("Dists:")
                    print(dist)
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
    args = parser.parse_args()
    main(args)
