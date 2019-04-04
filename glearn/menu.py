from glearn.utils.printing import print_tabular


# TODO - menu of experiments to train/evaluate/etc.


def menu():
    options = {}  # TODO - train/eval, debug/-, verbose/-, profile/-, random/-, ...version?
    option_count = 0

    from glearn.utils.config import list_configs
    configs = list_configs()
    experiments = {i + option_count: config for i, config in enumerate(configs)}

    tabs = {
        "Options": options,
        "Experiments": experiments,
    }
    print_tabular(tabs, grouped=True, show_type=False)

    selection = input(">> ")
    # TODO
    print(selection)
