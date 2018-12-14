import argparse
import json
import os

def process_config(cfg_dict):
    if "paths" in cfg_dict:
        for key in cfg_dict["paths"]:
            cfg_dict[key] = os.path.expanduser(cfg_dict["paths"][key])
        del cfg_dict["paths"]
    return cfg_dict

def parse_config(script_name=None):
    cfg_suffix = ".json"
    default_cfg_path = "./"
    default_cfg_file = os.path.join(default_cfg_path, script_name + cfg_suffix)
    # Turn off help, so we print all options in response to -h
    cfg_parser = argparse.ArgumentParser(add_help=False)
    cfg_parser.add_argument("-c", "--cfg_file", default=default_cfg_file,
                            help="Specify config file (json)", metavar="FILE")
    cfg_parser.add_argument("-n", "--exp_name", required=False,
                            help="Specify current experiment name")
    args, remaining_argv = cfg_parser.parse_known_args()

    if args.cfg_file:
        cfg_dict = json.load(open(args.cfg_file))
    defaults = process_config(cfg_dict["defaults"])
    parser = argparse.ArgumentParser(
        parents=[cfg_parser],
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,)
    parser.set_defaults(**defaults)
    if args.exp_name:
        cur_exp = process_config(cfg_dict[args.exp_name])
        parser.set_defaults(**cur_exp)  # override with cur_exp setting

    return parser