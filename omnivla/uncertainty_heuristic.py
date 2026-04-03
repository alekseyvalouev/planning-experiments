# We wish to compute uncertainty heuristic as follows:
# Measure uncertainty wrt goal. 
# Specify some “goal state”. The way that we will calculate uncertainty is by normalizing both actions to magnitude 1, then calculating mse between (o_curr, o_candidate) and (o_curr, o_end).
# The uncertainty is then the mse.


# omnivla will provide some wrapper for us to compute the action. 

# import omnivla stuff
import numpy as np
from OmniVLA.inference.run_omnivla import Inference, InferenceConfig, define_model

# modality mask is (language, vision, pose)
def setup():
        # Define models (VLA, action_head, pose_projector, processor, etc.)
    cfg = InferenceConfig()
    vla, action_head, pose_projector, device_id, NUM_PATCHES, action_tokenizer, processor = define_model(cfg)

    # Run inference
    inference = Inference(
        save_dir="./inference",
        lan_inst_prompt=lan_inst_prompt,
        goal_utm=goal_utm,
        goal_compass=goal_compass,
        goal_image_PIL=goal_image_PIL,
        action_tokenizer=action_tokenizer,
        processor=processor,
    )
    return inference

def produce_action(curr_img, goal_modal):
    actions = inference.get_action(curr_img, goal_modal["language"], goal_modal["vision"], goal_modal["pose"])
    return actions.float().cpu().numpy()

def compute_uncertainty(curr_img, goal_0, goal_1):
    actions_0 = produce_action(curr_img, goal_0)
    actions_1 = produce_action(curr_img, goal_1)    

    # Normalize both actions to magnitude 1
    actions_0_norm = actions_0 / np.linalg.norm(actions_0)
    actions_1_norm = actions_1 / np.linalg.norm(actions_1)

    # Calculate MSE between (o_curr, o_candidate) and (o_curr, o_end)
    mse = np.mean(np.square(actions_0_norm - actions_1_norm))

    return mse

