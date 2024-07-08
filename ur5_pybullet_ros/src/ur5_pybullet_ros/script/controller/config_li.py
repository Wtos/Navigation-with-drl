shared_variable = 0

def set_shared_variable(value):
    global shared_variable
    shared_variable = value

def get_shared_variable():
    global shared_variable
    return shared_variable