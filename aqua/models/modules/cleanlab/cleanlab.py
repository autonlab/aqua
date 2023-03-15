import cleanlab as cl

class CleanLab:
    def __init__(self, model):
        self.model = model

    def find_label_issues(self, data, label, **kwargs):
        wrapper_model = cl.classification.CleanLearning(self.model)
        label_issue_summary = wrapper_model.find_label_issues(data, label)
        return label_issue_summary['is_label_issue']