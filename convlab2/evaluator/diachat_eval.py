from convlab2.evaluator.evaluator import Evaluator

class DiachatEvaluator(Evaluator):
    def __init__(self):
        self.sys_da_array = []
        self.usr_da_array = []
        self.goal = {}
        self.cur_domain = ''
        self.booked = {}
        self.dbs = self.database.dbs