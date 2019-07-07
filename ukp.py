import sys

class ukp:
    capacity = 0
    p = []  # Profits Array
    w = []  # Weights Array

    def __init__(self, capacity, p, w):
        self.capacity = capacity
        self.p = list(p)
        self.w = list(w)

    def validate(self):
        if self.capacity < 0 or len(self.p) != len(self.w):
            print
            "Validation error"
            sys.exit()

class rowtwin:
    def __init__(self, obj, row):
        self.row = row
        self.obj = obj

def ukp_dno(ukp_obj):
    twins = []
    ukp_obj.validate()

    for i in range(0, len(ukp_obj.p)):
        tmp_row = int(ukp_obj.p[i]/ukp_obj.w[i])
        twins.append(rowtwin(i, tmp_row))

    twins.sort(key=lambda x: x.row, reverse=True)

    for twin in twins:
        print (str(twin.obj) + ":" + str(twin.row))

# main
instance = ukp(10, [5, 6, 7, 8], [1, 2, 3, 4])
ukp_dno(instance)
