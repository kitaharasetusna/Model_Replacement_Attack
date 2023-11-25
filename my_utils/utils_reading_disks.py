import yaml


def get_dict_from_yaml(path: str):
    with open(path, 'r') as f_:
        config_dict = yaml.safe_load(f_)
    return config_dict



if __name__ == "__main__":
    yaml_data = get_dict_from_yaml('../configs/test.yaml')
    print(yaml_data)
