TASK_ALIGNMENT_PROMPT = """
Grade the alignment between the plan (temporally ordered sequence of visited location descriptions) and the task (a description of steps to be taken). Your response should be a single integer between 1 and 100, representing how closely the plan aligns with the task. The steps taken in the plan must be in the same order as the associated instructions in the task. 
"""