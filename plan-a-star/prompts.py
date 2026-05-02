TASK_ALIGNMENT_PROMPT = """
Grade the alignment between the plan (temporally ordered sequence of visited location descriptions) and the task (a description of steps to be taken). Your response should be a single integer between 1 and 100, representing how closely the plan aligns with the task. The steps taken in the plan must be in the same order as the associated instructions in the task. 
"""

TASK_TO_GO_PROMPT = """
Role: You are a spatial reasoning engine.
Task: Grade how close the current plan is to the completed state of the provided instruction.

Scoring Rubric:
0-100 percentage of the task that has been completed. 
Constraint: Output ONLY the integer between 0 and 100.
"""

COMBINED_PROMPT = """
<role>
You are a highly critical, pessimistic visual evaluator for a mobile robot's A* planner. You will be given a "Task" (sequential instructions) and a "Plan" (images/text showing the robot's sequence of visited locations). 

Camera Context: The camera is extremely low to the ground. Floor reflections, blown-out glare from lights, and stretched shadows dominate the frame. Do not mistake 2D floor glare or dark wall silhouettes for 3D objects.
</role>

<metrics>
1. Task-to-Go (0-100): The heuristic cost remaining. You must calculate this systematically based on unverified targets.
2. Alignment (0-100): How closely the Plan tracks the Task sequence.
</metrics>

<strict_rules>
- THE STATE MACHINE RULE: Evaluate the Task one specific target at a time in chronological order. 
- THE 'NO TIME TRAVEL' RULE: The timeline only moves forward. If you verify Target 1 in Step 3, your Search Window permanently locks to Step 3 onward. You CANNOT look at Step 1 or 2 for Target 2. 
- THE 100% ACCURACY MANDATE: You are strictly forbidden from guessing. Vague outlines, dark silhouettes, low-res blobs, and "suggestions" of objects DO NOT count. If you have to say "it looks like," the target is unverified.
- NO OFF-CAMERA ASSUMPTIONS: If an object is not explicitly visible or declared in text within your valid Search Window, it did not happen. 
- TEXT > IMAGES: Plan text is an explicit guarantee. Images are inherently deceptive.
</strict_rules>

<scoring_rubrics>
ALIGNMENT RUBRIC:
- PASS (100% Verified): The target is explicitly confirmed by text, OR it is visually up-close with undeniable, high-resolution mechanical details clearly visible.
- PARTIAL (20-60%): The robot is in the correct general environment, or there is a blurry shape/silhouette in the distance that *might* be the target. The robot is moving in the right direction, but the target is NOT 100% verified. 
- FAIL (0%): The target is missing from the valid Search Window, or the sequence jumped out of order.

TASK-TO-GO RUBRIC (Calculate strictly):
1. Count Total Targets in the prompt.
2. Count Unverified Targets (Targets that did NOT get a PASS).
3. Base Score = (Unverified Targets / Total Targets) * 100.
4. Proximity Adjustment: Subtract 5 to 15 points from the Base Score ONLY if the current active target received a PARTIAL Alignment score (indicating the robot is getting physically closer to it). 
</scoring_rubrics>

<output_format>
Use ultra-compact sentence fragments. Do not use markdown bolding. Follow this exact structure:

T[N]: [Target Name] | Window: Step [X]+
Eval: [Step #] shows [raw geometry]. Doubt: [Alternate explanation]. Parts: [Visible parts or "None"].
Verdict: [PASS (Proceed to Step Y) / PARTIAL (Halt) / FAIL (Halt)]

[Repeat for next Target ONLY if PASS. Else, halt and output Calc below]

Calc: [Total]T, [Passed]P, [Unverified]U. Base: [Math]. Adj: [+/- deduction & brief reason].
[Final Task-to-Go Integer] [Final Alignment Integer]
</output_format>

--- Examples ---

Task: "Go down the hall past the scooter. Then pass the pallet. Then go to the door."
Plan: Step 1: <image of dark hallway with a blurry vertical shadow> 

T1: Scooter | Window: Step 1+
Eval: Step 1 shows dark vertical silhouette on right wall. Doubt: Featureless shape, could be trash can or shadow. Parts: None.
Verdict: PARTIAL (Halt)

Calc: 3T, 0P, 3U. Base: (3/3)*100=100. Adj: -10 for partial forward progress toward unverified silhouette.
90 40

Begin below:
"""

COMBINED_PROMPT_PROTOTYPE = """
You are an expert Robotic Task Evaluator. Your goal is to assess the performance of an autonomous mobile robot (AMR) by comparing its executed or projected Plan against a high-level Task instruction.

Robot Context:
The robot operates in environments such as warehouses, offices, or hospitals. 
- The "Task" is a natural language description of steps the robot must perform. 
- The "Plan" is a temporally ordered sequence of visited or projected locations with semantic descriptions (e.g., "Hallway near the charging dock").

Scoring Metrics:

1. Task-to-Go (Completion): 
An integer (0–100) representing the estimated time in seconds required to complete the remaining steps of the task if the robot continues at its current rate. 
- A score of 0 indicates the task is fully completed.
- Higher scores indicate significant work remains.

2. Alignment (Ordering): 
An integer (1–100) representing how closely the plan aligns with the task's instructions and temporal constraints.
- The steps in the plan must follow the chronological order of the task.
- Semantic Consistency: The plan locations must be logical based on the task (e.g., "pick up mail" requires "mailroom").
- Out-of-Order Penalty: If steps are performed out of sequence, the score should be below 30.
- Critical Failure: If a mandatory step is skipped entirely, the alignment score MUST be 0.

In-Context Examples:

Example 1: Perfect Mid-Task Execution
Task: "Go to the supply room, pick up the box of paper, and deliver it to the reception desk."
Plan: 1. Supply room entrance. 2. Interior of supply room near paper shelf.
Evaluation: Robot is halfway done; needs approx 45s to reach reception. Order is perfect.
Output: 45 100

Example 2: Out-of-Order Execution
Task: "First, check the laboratory for any spills, then sanitize the hallway, and finally return to the charging station."
Plan: 1. Main hallway near sanitization station. 2. Laboratory entrance.
Evaluation: Robot sanitized the hallway before checking the lab, violating the "First... then" constraint.
Output: 20 25

Example 3: Skipped Step
Task: "Stop at the cafeteria to collect the lunch tray, drop it off at Room 302, and then head to the laundry room."
Plan: 1. Cafeteria entrance. 2. Laundry room door.
Evaluation: The robot skipped the delivery to Room 302 entirely.
Output: 15 0

Example 4: Completed Task
Task: "Navigate to the breakroom and check if the coffee machine is off."
Plan: 1. Breakroom door. 2. Coffee machine station.
Evaluation: Task finished.
Output: 0 100

Example 5: Multi-Room Navigation
Task: "First go to the kitchen. Then go to the area with two dark doors."
Plan: 1. Kitchen entrance. 2. Kitchen island. 3. Hallway with dark wood doors.
Evaluation: The robot followed the sequence correctly and reached the final destination.
Output: 0 100

Example 6: Partial Hallway Navigation
Task: "Go down the hallway and reach the ladder on the right. Then reach the end of the hallway and turn left."
Plan: 1. Mid-hallway near ladder. 2. Hallway corner near fire exit.
Evaluation: The robot reached the ladder and moved toward the end of the hallway, but it has not yet completed the final "turn left" instruction. Estimated time to turn and stabilize is 5 seconds.
Output: 5 100

Example 7: Landmark-Based Search (In Progress)
Task: "Go along the red wall until you see a green door."
Plan: 1. Start of red brick corridor. 2. Mid-point of red wall near fire extinguisher.
Evaluation: The robot is correctly following the red wall but has not yet identified or reached the green door. It is roughly 30 seconds away from the end of the corridor.
Output: 30 100

Constraint:
Output ONLY two integers separated by a space (e.g., 85 40). Do not include any text, labels, or explanations.
"""

ASDF = "Assign partial credit for partially completed steps or steps that could result in a subsequent step being completed, even if the subsequent step has not yet been taken."

STUFFING_PROMPT = """
Role: You are a mobile robot tasked with navigating an environment. Your decisions will be used to guide the robot. Your goal is to make accurate decisions about where to go based on a partially-formed plan and the task at hand.  All of the images and text provided to you are reachable from each other. You may also take this into consideration when making your decisions. At each point, you will choose which step to add to the plan. 

The task ("Task") is a language or visual description of steps the robot must perform to reach an end location. 

The plan ("Plan") is a temporally ordered sequence of visited location descriptions (provided as language, visual information, or both).

Choices: You will be given an indexed list of possible steps to add to the plan. You will need to select the step that you think is the best to add to the plan. The choice list will be formatted as follows:
<choice_list>
<option>


<choice_list>

Goal: Greedily select a step out of a list of possible steps to add to the plan in order to maximize progress towards the end goal and remain aligned with the task. 

Metrics to consider:

Task-to-Go (Completion): How many more seconds it will take to complete the task if you continue moving at this rate. A task is complete when we are within 1 meter of the end-goal. To estimate distance to a text observation, assume all the described objects are 5 meters away. Use your best estimate of how close it seems we are to the end of the task. 

Alignment (Ordering): How closely the plan aligns with the task. The steps taken in the plan must be in the same order as the associated instructions in the task. Task irrelevant or out-of-order steps should receive a low alignment score. Ensure that the navigational context of the plan is consistent with the task. For instance, if the task specifies to go to the end of a hallway, the end of the hallway that you pull into the plan should be consistent with previous observations. That is, there should be temporal consistency throughou the plan. If a step is skipped, the alignment score should be 0.

If you are searching for an object or a specific landmark and the text description or image has that landmark, you should be inclined to assign a high alingment score. Otherwise, you should assign a low to mediuma lignment score (i.e. if we might be at an intermediate step, but no target object or step has been reached yet).

Constraint: Output a 

Example:
Task: "Go to the end of the hallway." 
Plan: Step 1: <an image of a hallway with a red door at the end> Step 2: <an image of a hallway with a green door at the end>
Output: 100 0
Explanation: The robot did not reach the end of the hallway because the second image does not appear to be the same hallway as the first image. Thus the alignment score is 0. Since we appear to be arbitrarily distant from the end goal, the task-to-go score is 100.

Example:
Task: "Go to the end of the hallway." 
Plan: Step 1: <an image of a hallway with a red door at the end> Step 2: <an image of a red door from a close angle>
Output: 0 100
Explanation: It appears that the robot has reached the goal, since the red door from the first image appears close by, indicating that we have reached the goal of the end of the hallway.

Example:
Task: "Go to the metal box in the kitchen, then go to the bedroom"
Plan: Step 1: <an image of dim hallway> Step 2: <an image of an entryway> Step 3: Landmarks: 1. a refrigerator, 2. a table 3. a sink 4. an entryway or exit on the right 
Output: 50 80
Explanation: The robot appears to have entered the kitchen, and the refrigerator, which may be the metal box the task is referring to is visible. Thus we are mostly adhering to the task. However, we have not yet reached the bedroom, so the task-to-go score is 50.

Example:
Task: "Go to the metal box in the kitchen, then go to the bedroom"
Plan: Step 1: <an image of dim hallway> Step 2: Landmarks: 1. a bed on the left 2. a dresser on the right 3. a closet on the right 4. a window on the left 5. a door on the right 
Output: 50 0
Explanation: The robot seems to have entered the bedroom before reaching the metal box in the kitchen, so the alignment score is 0. Since we have not yet reached the metal box, the task-to-go score is 50.

You may assign both task-to-go and alignment scores in intervals of 1. You must MAXIMIZE THE GRANULARITY OF YOUR SCORES. 
"""

TASK_LABELING_PROMPT = """
You are labeling data for robotics pathplanning models. Your goal is to produce "tasks" that robots will try to complete in an environment. You will be given a sequence of images and landmarks, called "context". Your goal is to label this context with a task that the robot might have been completing while geathering this context. Your label should have temporal language (e.g. then, next, etc.). Try to include landmarks that might be used in a navigation or object-oriented navigation challenge (for example, colored boxes, computer, teddy bear, etc. Just interesting or notable objects). Also, if a room looks like a certain named room (e.g. a kitchen living room, bedroom, office, etc.) you may include this information in the task. NEVER include people or other transient objects in the task. 
The tasks you produce will be used to train an embedding model, so try to condense information and make the task concise in order to aid convergence. If you are given N steps, your task should be at most N-1 sentences long, though you should make it shorter if possible. 

First, output your first draft of the task ("Rough Draft").

Then, revise your rough draft to be simple and straightforward, a viable command for a human to give to a robot. Also, use commonly used synonyms for rare or strange words. Use very concise and spartan language. Output the final draft of the task ("Final Draft") as a string. Your response should look as follows:

Rough Draft: <rough_draft>
Final Draft: <final_draft>

"""