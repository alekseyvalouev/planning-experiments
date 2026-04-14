import sys, os
import math

import numpy as np
from PIL import Image
import utm

import torch

OMNIVLA_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "OmniVLA"))
sys.path.insert(0, OMNIVLA_ROOT)

from inference.run_omnivla import (
    Inference,
    InferenceConfig,
    define_model,
)


def setup_omnivla():
    """Load all OmniVLA models and return the Inference object plus model artifacts."""
    pose_goal = False
    satellite = False
    image_goal = True
    lan_prompt = False

    lan_inst_prompt = "move toward blue trash bin"
    goal_lat, goal_lon, goal_compass = 37.8738930785863, -122.26746181032362, 0.0
    goal_utm = utm.from_latlon(goal_lat, goal_lon)
    goal_compass = -float(goal_compass) / 180.0 * math.pi
    goal_image_PIL = Image.open(os.path.join(OMNIVLA_ROOT, "inference", "goal_img.jpg")).convert("RGB")

    cfg = InferenceConfig()
    vla, action_head, pose_projector, device_id, num_patches, action_tokenizer, processor = define_model(cfg)

    inference = Inference(
        save_dir=os.path.join(OMNIVLA_ROOT, "inference"),
        lan_inst_prompt=lan_inst_prompt,
        goal_utm=goal_utm,
        goal_compass=goal_compass,
        goal_image_PIL=goal_image_PIL,
        action_tokenizer=action_tokenizer,
        processor=processor,
        vla=vla,
        action_head=action_head,
        pose_projector=pose_projector,
        device_id=device_id,
        num_patches=num_patches,
        pose_goal=pose_goal,
        satellite=satellite,
        image_goal=image_goal,
        lan_prompt=lan_prompt,
    )

    return inference

def infer_action_V_V(inference, start_image, end_image): 
    #images passed as PIL images

    inference.pose_goal = False
    inference.image_goal = True
    inference.lan_prompt = False
    inference.satellite = False
    inference.goal_image_PIL = end_image

    actions = inference.run_omnivla(start_image)
    return actions

def infer_action_V_VL(inference, start_image, end_image, end_prompt): 
    #images passed as PIL images

    inference.pose_goal = False
    inference.image_goal = True
    inference.lan_prompt = True
    inference.lan_inst_prompt = end_prompt
    inference.satellite = False
    inference.goal_image_PIL = end_image

    actions = inference.run_omnivla(start_image)
    return actions

def infer_action_V_L(inference, start_image, end_prompt): 
    #images passed as PIL images

    inference.pose_goal = False
    inference.image_goal = False
    inference.lan_prompt = True
    inference.lan_inst_prompt = end_prompt
    inference.satellite = False

    actions = inference.run_omnivla(start_image)
    return actions

def calculate_action_distance(action1, action2):
    return torch.square(torch.linalg.norm(action1) - torch.linalg.norm(action2)).mean()

if __name__ == "__main__":
    start_image = Image.open("imgs/bww1-0-t20.png").convert("RGB")
    unrelated_image = Image.open("imgs/soda-0-t20.png").convert("RGB")
    end_image = Image.open("imgs/bww1-0-t100.png").convert("RGB")
    inference = setup_omnivla()

    actions_base = infer_action_V_V(inference, start_image, start_image).cpu()
    acitons_language = infer_action_V_L(inference, start_image, "Go towards the wall on the left").cpu()
    actions_distant = infer_action_V_V(inference, start_image, unrelated_image).cpu()
    action_end = infer_action_V_V(inference, start_image, end_image).cpu()
    print(actions_base)
    print(actions_distant)
    print(f"MSE between base and distant actions: {calculate_action_distance(actions_base, actions_distant)}")
    print(f"MSE between end and language actions: {calculate_action_distance(action_end, acitons_language)}")
    print(f"MSE between base and end actions: {calculate_action_distance(actions_base, action_end)}")
    print(f"MSE between distant and end actions: {calculate_action_distance(actions_distant, action_end)}")