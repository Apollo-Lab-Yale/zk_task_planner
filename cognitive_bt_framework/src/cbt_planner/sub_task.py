class SubTask(object):
    def __init__(self, name, condition):
        self.name = name
        self.children = []
        self.failed_plans = []
        self.successful_plans = []
        self.condition = ""