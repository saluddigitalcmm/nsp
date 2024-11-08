import json
import matplotlib.pyplot as plt
import os
import numpy as np

class CrossValVisualizer:
    def __init__(self,results_location):
        self.scores_total = {}
        for filename in os.listdir(results_location):
            if filename.endswith(".json"):
                estimator_name = filename.split('.')[0]
                self.scores_total[estimator_name] = {}
                with open(results_location + filename, "r") as read_file:
                    data = json.load(read_file)
                    self.scores_total[estimator_name] = data
                self.scores = [score['test_f2_True'] for score in self.scores_total.values()]
                self.models = [model for model in self.scores_total.keys()]
                self.sorted_data = [(model_name,scores,np.mean(scores)) for model_name,scores in zip (self.models, self.scores)]
                self.sorted_data.sort(key=lambda tup: tup[2])
                self.models, self.scores, _ = zip(*self.sorted_data)
    def plot(self,figure_location):
        plt.boxplot(self.scores, labels=self.models)
        plt.title('Model Performances')
        plt.xlabel('Model')
        plt.ylabel('F2 Score')
        plt.xticks(rotation=45)
        plt.tight_layout()
        if os.path.exists(os.path.dirname(figure_location)) == False:
            os.makedirs(os.path.dirname(figure_location))
        plt.savefig(figure_location)