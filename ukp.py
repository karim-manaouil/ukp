import sys
import time

# This is the class that contains the information
# of an instance : capacity, weights list and profits list
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

# This is useful when you need a pair of
# (value, object) in your algorithm
class rowtwin:
    def __init__(self, obj, row):
        self.row = row
        self.obj = obj

# This is what all the algorithms must return
# the total profit, the list of taken objects
# and the list that contains how many times
# each object has been taken
class ukp_solution:
    def __init__(self):
        self.total = 0
        self.taken = []
        self.ttimes = []

# This method inserts an object into the solutions structure
def ukp_select_object (ukp_solution_o, object):
        if not object in ukp_solution_o.taken:
            ukp_solution_o.taken.append(object)
            ukp_solution_o.ttimes.insert(object, 1)
        else :
            ukp_solution_o.ttimes[object] += 1


# Density ordered heuristic, it takes a ukp object as parameter
# and return a ukp_solution
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

    ukp_sol_o = ukp_solution()

    while current_capacity > 0 and cont:
        if (ukp_obj.w[current_obj] > current_capacity):
            if ++current_obj_i < len(ukp_obj.p):
                current_obj = twins[current_obj_i].obj
            else:
                cont = False
            continue

        ukp_select_object(ukp_sol_o, current_obj)
        ukp_sol_o.total += ukp_obj.p[current_obj]
        current_capacity -= ukp_obj.w[current_obj]

    return ukp_sol_o

# total value heuristic
def ukp_tv(ukp_object):
    ukp_object.validate()
    cc = ukp_object.capacity # left capacity in each iteration

    rem_objects = list(range(0, len(ukp_object.p))) # Remaining objects
    ukp_sol_o = ukp_solution()

    while (cc > 0 and len(rem_objects) > 0):
        max_metric = 0; selected = -1
        for object in rem_objects:
            metric = ukp_object.p[object] * int(cc/ukp_object.w[object])
            if metric > max_metric:
                max_metric = metric
                selected = object

        cc = cc - ukp_object.w[selected]
        if cc < 0:
            break

        ukp_select_object(ukp_sol_o, selected)
        ukp_sol_o.total += ukp_object.p[selected]
        rem_objects.remove(selected)

    return ukp_sol_o


def execute_instance(type, ukp_o):
    start = time.localtime()

    if type == "ukp_dno":
        solution = ukp_dno(ukp_o)
    elif type == "ukp_tv":
        solution = ukp_tv(ukp_o)
    else :
        return

    end = time.localtime()

    print (type)
    print ("Executed in " + str(end.tm_sec - start.tm_sec) + " secs")
    print ("Total profit = " + str(solution.total))
    print ("chosen objects:times")
    for i in range(0, len(solution.taken)):
        print(str(solution.taken[i]) + ":" + str(solution.ttimes[i]))


# main
instance = ukp(10, [5, 6, 7, 8], [1, 2, 3, 4])

execute_instance("ukp_dno", instance)
print ("\n")
execute_instance("ukp_tv", instance)