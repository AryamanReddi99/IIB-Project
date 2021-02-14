import numpy as np
from general import *

class evaluate_q_table():
    # class which evaluates whether a q-table has reached the optimal policy
    def __init__(self,i,n,q_table):
        self.i=i
        self.n=n
        self.q_table=q_table
        self.row_evaluations = None
    def evaluate_row(self,row_i):
        a = self.n
        while a > row_i :
            prev_a = a
            a -= self.i+1
        if a == row_i:
            return True
        if row_i == 0 or row_i == 1:
            if all(item == 0 for item in self.q_table[row_i]):
                return True
        optimal_play = prev_a - row_i
        if self.q_table[row_i].argmax() == optimal_play-1:
            return True
        else:
            return False
    def evaluate_q_table(self):
        self.row_evaluations = [self.evaluate_row(row_i) for row_i in range(self.n+1)]
        if all(item for item in self.row_evaluations):
            return True
        else:
            return False
    def faulty_rows(self):
        return [i for i,state in enumerate(self.row_evaluations) if not state]
    def abs_vals(self):
        return np.sum(np.absolute(self.q_table))
    def rows_abs_vals(self):
        row_abs = []
        for row_i in range(0,self.n+1):
            row = self.q_table[row_i]
            row_abs.append(np.sum(np.absolute(row)))
        return row_abs


def main():
    eg_table = np.array([[10,2,3],[2,3,4],[3,20,4],[10,2,3],[2,3,4],[3,2,4],[1,28,3],[29,3,4],[30,2,4]])
    evaluator = evaluate_q_table(3,9,eg_table)
    if not evaluator.evaluate_q_table():
        print(evaluator.faulty_rows())
    print(evaluator.row_evaluations)
    print(evaluator.abs_vals())
    print("done")

if __name__ == "main":
    main()
