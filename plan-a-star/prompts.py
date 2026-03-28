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
Grade the alignment between the plan (temporally ordered sequence of visited location descriptions) and the task (a description of steps to be taken).

Scoring Metrics:

Task-to-Go (Completion): An integer (0–100) representing the percentage of the task steps that have been successfully completed in the plan. Assign partial credit for partially completed steps.

Alignment (Ordering): An integer (1–100) representing how closely the plan aligns with the task. The steps taken in the plan must be in the same order as the associated instructions in the task. Task irrelevant or out-of-order steps should receive a low alignment score. Ensure that the semantic sof the iamge or language provided in the plan is consistent with the task as well.

Constraint: Output ONLY two integers separated by a space (e.g., 85 40). The first integer is Task-to-Go, and the second is Alignment.
"""