import PIL.Image as Image
import math

def plant_triggers(inputs, config: dict = None):
    poisoned_portion, pos = config["portion_pois"], config["pos"]
    poisoned_num = math.ceil(inputs.shape[0] * poisoned_portion)

    
    poisoned_inputs = inputs[:poisoned_num].clone()
    poisoned_inputs[:, :, pos:, pos:] = trigger
    clean_inputs = inputs[poisoned_num:]

    


if __name__ == "__main__":
    trigger_path = 'triggers/phoenix.png'
    print(trigger_path)
