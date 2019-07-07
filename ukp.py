import sys
import time

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

class ukp_solution:
    def __init__(self, taken, ttimes, total):
        self.total = total
        self.taken = taken
        self.ttimes = ttimes

def ukp_dno(ukp_obj):
    twins = []
    ukp_obj.validate()

    for i in range(0, len(ukp_obj.p)):
        tmp_row = int(ukp_obj.p[i]/ukp_obj.w[i])
        twins.append(rowtwin(i, tmp_row))

    twins.sort(key=lambda x: x.row, reverse=True)
    # for twin in twins:
    #     print (str(twin.obj) + ":" + str(twin.row))

    current_capacity = ukp_obj.capacity
    total_profit = 0
    current_obj_i = 0
    current_obj = twins[current_obj_i].obj
    cont = True

    taken = []
    ttimes = []

    while current_capacity > 0 and cont:
        if (ukp_obj.w[current_obj] > current_capacity):
            if ++current_obj_i < len(ukp_obj.p):
                current_obj = twins[current_obj_i].obj
                continue
            else:
                cont = False
                continue
        if not current_obj in taken:
            taken.append(current_obj)
            ttimes.insert(current_obj, 1)
        else :
            ttimes[current_obj] += 1

        total_profit += ukp_obj.p[current_obj]
        current_capacity -= ukp_obj.w[current_obj]

    return ukp_solution(taken, ttimes, total_profit)


# main
instance = ukp(10, [5, 6, 7, 8], [1, 2, 3, 4])

start = time.localtime()
solution = ukp_dno(instance)
end = time.localtime()

print ("Executed in " + str(end.tm_sec - start.tm_sec) + " secs")
print ("Total profit = " + str(solution.total))
for i in range(0, len(solution.taken)):
    print (str(solution.taken[i]) + ":" + str(solution.ttimes[i]))
