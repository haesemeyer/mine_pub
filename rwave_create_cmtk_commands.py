
import argparse
import os
from os import path
from typing import Any


class CheckArgs(argparse.Action):
    """
    Check our command line arguments for validity
    """
    def __init__(self, option_strings, dest, nargs=None, **kwargs):
        if nargs is not None:
            raise ValueError("nargs not allowed")
        super().__init__(option_strings, dest, **kwargs)

    def __call__(self, parser, namespace, values: Any, option_string=None):
        if self.dest == 'folder':
            if not path.exists(values):
                raise argparse.ArgumentError(self, "Specified directory does not exist")
            if not path.isdir(values):
                raise argparse.ArgumentError(self, "The destination is a file but should be a directory")
            setattr(namespace, self.dest, values)
        else:
            raise Exception("Parser was asked to check unknown argument")


def create_registration_command(crop, outfile, template, infile):
    return f"cmtk registration --dofs 6,9 --verbose --initxlate --ncc --exploration 30 --accuracy 0.8 " \
           f"--auto-multi-levels 4 --threads 6 --crop-index-ref {crop} -o {outfile} {template} {infile}"


def create_warp_command(crop, outfile, template, init, infile):
    return f"cmtk warp --grid-spacing 80 --refine 3 --exploration 26 --coarsest 8 --match-histograms --accuracy 0.8 " \
           f"--verbose --threads 6 --jacobian-weight 8e-5 --energy-weight 5e-1 --crop-index-ref {crop} -o " \
           f"{outfile} --initial {init} {template} {infile}"


def create_reformat_command(template, infile, transform, outfile):
    return f"cmtk reformatx --outfile {outfile} --floating {infile} {template} {transform}"


if __name__ == '__main__':

    a_parser = argparse.ArgumentParser(prog="general_zfishexp_plots",
                                       description="Will perform general overview data analysis of zebrafish experiment"
                                                   " groups that have been ANN fitted and Taylor decomposes.")
    a_parser.add_argument("-f", "--folder", help="Path to folder with experiment hdf5 files", type=str, default="",
                          action=CheckArgs)

    crop_dict = {
        "FBd": "\"0,645,0,768,1380,65\"",
        "FBv": "\"0,645,40,768,1380,119\"",

        "MBd": "\"0,290,0,768,975,65\"",
        "MBv": "\"0,290,40,768,975,119\"",

        "HBd": "\"0,0,0,768,790,65\"",
        "HBv": "\"0,0,40,768,790,119\""
    }

    args = a_parser.parse_args()

    data_folder = args.folder

    file_list_ch0 = sorted([f for f in os.listdir(data_folder) if ".nrrd" in f and "Ch0" in f and "warped" not in f])
    file_list_ch1 = sorted([f for f in os.listdir(data_folder) if ".nrrd" in f and "Ch1" in f and "warped" not in f])

    template_path = "/media/ac/Data01/refbrain/H2B6s_Template_Final.nrrd"

    command_file = open(path.join(data_folder, "cmtk_commands.sh"), 'w')
    command_file.write('#!/bin/bash\n')
    for f_ch0, f_ch1 in zip(file_list_ch0, file_list_ch1):
        registration_source = path.join(data_folder, f_ch0)
        region_code = f_ch0[f_ch0.find("vlgut_")+6:f_ch0.find("vlgut_")+9]
        crop_command = crop_dict[region_code]
        registration_out = path.join(data_folder, f_ch0[:f_ch0.find(".nrrd")] + "_affine.xform")
        warp_out = path.join(data_folder, f_ch0[:f_ch0.find(".nrrd")] + "_ffd5.xform")
        final_out_ch0 = path.join(data_folder, f_ch0[:f_ch0.find(".nrrd")] + "_warped.nrrd")
        final_out_ch1 = path.join(data_folder, f_ch1[:f_ch1.find(".nrrd")] + "_warped.nrrd")
        ch0_in = path.join(data_folder, f_ch0)
        ch1_in = path.join(data_folder, f_ch1)
        command_file.write(create_registration_command(crop_command, registration_out, template_path, ch0_in) + '\n')
        command_file.write(create_warp_command(crop_command, warp_out, template_path, registration_out, ch0_in) + '\n')
        command_file.write(create_reformat_command(template_path, ch0_in, warp_out, final_out_ch0) + '\n')
        command_file.write(create_reformat_command(template_path, ch1_in, warp_out, final_out_ch1) + '\n')
        command_file.write("\n")


