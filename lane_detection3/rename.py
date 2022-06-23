import os
import shutil

for fname in os.listdir('Archive'):
    string_list = fname.split('_')
    if string_list[-1][0].isdigit():
        date = string_list[-1].split('.')
        date[0], date[1] = date [1], date[0]
        new_name = '_'.join(string_list[:-1]) + '_' + '.'.join(date)
        os.rename(f'Archive/{fname}', f'Archive/{new_name}')
