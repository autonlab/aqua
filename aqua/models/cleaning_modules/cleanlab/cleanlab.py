import cleanlab as cl

class CleanLab:
    def __init__(self, model):
        self.model = model

    def find_label_issues(self, data_aq, **kwargs):
        data, label = data_aq.data, data_aq.labels
        wrapper_model = cl.classification.CleanLearning(self.model)
        label_issue_summary = wrapper_model.find_label_issues(data, label, clf_kwargs={'data_kwargs':data_aq.kwargs})
        return label_issue_summary['is_label_issue']