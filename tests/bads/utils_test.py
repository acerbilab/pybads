
from pybads.bads.options import Options
def load_options(D, path_dir):
    # load basic and advanced options and validate the names
    pybads_path = path_dir
    basic_path = pybads_path + "/option_configs/basic_bads_options.ini"
    options = Options(
        basic_path,
        evaluation_parameters={"D": D},
        user_options=None,
    )
    advanced_path = (
        pybads_path + "/option_configs/advanced_bads_options.ini"
    )
    options.load_options_file(
        advanced_path,
        evaluation_parameters={"D": D},
    )
    options.validate_option_names([basic_path, advanced_path])
    return options