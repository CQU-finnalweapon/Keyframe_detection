

def get_task_config(name: str):
    if name == 'can':
        TASK_NAME = "robomimic_transfer_can"
        FOLDER_NAME = "can"
        OBJECT_LIST = ['can']
        ROBOT_LIST = ['robot']
        SURFACE_LIST = ['light-colored table tray', 'dark grid table tray']
        DEMO_ID = 0
        TIME_LENGTH = 5.9
    elif name == 'square':
        TASK_NAME = "robomimic_square_hole_in_peg"
        FOLDER_NAME = "square"
        OBJECT_LIST = ['brown square hole', 'brown square peg', 'grey cylindrical peg']
        ROBOT_LIST = ['robot']
        SURFACE_LIST = ['white table']
        DEMO_ID = 0
        TIME_LENGTH = 6.35
    elif name == 'tool_hang':
        TASK_NAME = "robomimic_hang_up_tool"
        FOLDER_NAME = "tool_hang"
        OBJECT_LIST = ['wooden base with a rod', 'black-handled stick', 'metal hook']
        ROBOT_LIST = ['robot']
        SURFACE_LIST = ['white table']
        DEMO_ID = 0
        TIME_LENGTH = 34
    else:
        raise NotImplementedError(f"Task {name} not implemented.")
    return [TASK_NAME, OBJECT_LIST, ROBOT_LIST, SURFACE_LIST, DEMO_ID, TIME_LENGTH, FOLDER_NAME]