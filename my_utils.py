import PIL.Image as Image
import math
import copy

def plant_triggers(inputs, trigger, config: dict):
    poisoned_portion, pos, device = config["portion_pois"], config["pos"], config['device']
    poisoned_num = math.ceil(inputs.shape[0] * poisoned_portion)

    
    poisoned_inputs = inputs[:poisoned_num].clone()
    poisoned_inputs[:, :, pos:, pos:] = copy.deepcopy(trigger)
    clean_inputs = inputs[poisoned_num:]
    return poisoned_inputs[:poisoned_num].to(device), clean_inputs.to(device)

    


if __name__ == "__main__":
    trigger_path = 'triggers/phoenix.png'
    print(trigger_path)
