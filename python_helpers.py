import os

def get_project_root_dir():
    return os.path.dirname(os.path.abspath(__file__))

def get_main_data_dir():
    return os.path.join(get_project_root_dir(), 'data')

def print_with_indent(*to_print, n_indent: int, **kwargs):
    print('\t' * n_indent, end='')
    print(*to_print, **kwargs)