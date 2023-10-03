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
from threading import Lock

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
subgoal_image = None
subgoal_lock = Lock()

# Load the model using CUDA
device = torch.device("cuda")

def image_callback(msg):
    obs_img = msg_to_pil(msg)
    if len(context_queue) < model_params["context"] + 1:
        context_queue.append(obs_img)
    else:
        context_queue.pop(0)
        context_queue.append(obs_img)

def goal_callback(msg):
    with subgoal_lock:
        subgoal_image = msg_to_pil(msg)
    rospy.loginfo("Received new subgoal!")

def check_goal_reached(threshold, dist):
    raise NotYetImplemented()

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
    subgoal_image_sub = rospy.Subscriber("/gnm/subgoal", Image, goal_callback, queue_size=1)
    waypoint_pub = rospy.Publisher("/gnm/waypoint", Float32MultiArray, queue_size=1)
    goal_pub = rospy.Publisher("/gnm/reached_goal", Bool, queue_size=1)

    while not rospy.is_shutdown():
        if len(context_queue) > 1:
            transf_goal_img = None
            with subgoal_lock:
                if subgoal_image:
                    transf_goal_img = transform_images(subgoal_image, model_params["image_size"]).to(device)

            if transf_goal_img is not None and len(context_queue) > model_params["context"]:
                curr_im = np.asarray(context_queue[-1])
                transf_obs_img = transform_images(context_queue, model_params["image_size"]).to(device)
                start = time.time()
                dist, waypoint = model(transf_obs_img, transf_goal_img) 
                dist = to_numpy(dist[0])
                waypoint = to_numpy(waypoint[0])

                if check_goal_reached(args.close_threshold, dist):
                    rospy.loginfo("Reached subgoal!")
                    goal_pub.publish(True)
                    with subgoal_lock:
                        subgoal_image = None
                else: 
                    rospy.loginfo("*** Time: " + str(time.time() - start) + " | " + str(dist))
                    waypoint_msg = Float32MultiArray()
                    chosen_waypoint = waypoint[args.waypoint]
                    if model_params["normalize"]:
                        chosen_waypoint[:2] *= (MAX_V / RATE)
                    waypoint_msg.data = chosen_waypoint
                    waypoint_pub.publish(waypoint_msg)
        else:
            rospy.loginfo("Waiting for sensor data...")

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
        "--close-threshold",
        "-t",
        default=3,
        type=int,
        help="""temporal distance within the next node in the topomap before 
        localizing to it (default: 3)""",
    )
    args = parser.parse_args()
    main(args)
