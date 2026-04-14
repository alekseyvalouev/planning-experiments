# Once we have tasks, plans, and heuristics, we can build a dataset.
# Two options for building the dataset:
# 1. Use task + starting obs as query ex. (T, [o1, o2, o3], H). Q = o1 + Task, R = o3, Similarity = H
#   We call this non-recursive
# 2. Use task + second-last obs as query ex. (T, [o1, o2, o3], H). Q = o2 + Task, R = o3, Similarity = H
#   We call this recursive 
# Plan length must be at least 2. 
# TFDS dataset format


def DatasetBuilder:
    def __init__(self, graph_file):
        self.graph_file = graph_file
        self.task_builder = TaskBuilder(graph_file)
        self.data = []
    
    def _generate_task(self):
        path, task = self.task_builder.generate_task()
        if path is None:
            print("No path found. Generating new task.")
            return self._generate_task()
        return path, task
    
    def _generate_plan(self, path):
        pass
