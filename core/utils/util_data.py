import logging
from typing import List, Dict
import json
import pandas
import pickle
import os


class UtilData:
    def __init__(self):
        pass

    @staticmethod
    def read_raw_jsonl_file(file_name, verbose=True) -> List[dict]:
        if verbose:
            logging.info("reading jsonl from: {}".format(file_name))
        with open(file_name, 'r', encoding='utf-8') as file:
            file_content = [json.loads(line) for line in file]
        return file_content

    @staticmethod
    def read_raw_json_file(file_name, verbose=True) -> dict:
        if verbose:
            logging.info("reading json from: {}".format(file_name))
        with open(file_name, 'r', encoding='utf-8') as file:
            data = json.load(file)
        return data

    @staticmethod
    def read_raw_text_file(file_name, verbose=True) -> list:
        if verbose:
            logging.info("reading jsonl from: {}".format(file_name))
        with open(file_name, 'r', encoding='utf-8') as file:
            file_content = [line.strip() for line in file]
        return file_content

    @staticmethod
    def write_json_file(file_name, data, verbose=True):
        if verbose:
            logging.info("writing json to: {}".format(file_name))
        with open(file_name, 'w', encoding='utf-8') as file:
            json.dump(data, file, ensure_ascii=False, indent=4)

    @staticmethod
    def write_jsonl_file(file_name, data, verbose=True):
        if verbose:
            logging.info("writing jsonl to: {}".format(file_name))
        with open(file_name, 'w', encoding='utf-8') as file:
            for line in data:
                file.write(json.dumps(line, ensure_ascii=False))
                file.write('\n')

    @staticmethod
    def write_jsonl_file_line_error_catching(file_name, data, default_line=None, verbose=True):
        # if default_line is None, then ignore
        if verbose:
            logging.info("writing jsonl to: {}".format(file_name))
        with open(file_name, 'w', encoding='utf-8') as file:
            count = 0
            for line in data:
                try:
                    file.write(json.dumps(line, ensure_ascii=False))
                except UnicodeEncodeError:
                    logging.error('UnicodeEncodeError at file {} at line {}'.format(file_name, count))
                    try:
                        file.write(json.dumps(line, ensure_ascii=False).encode('utf-8', 'surrogateescape').decode('utf-8', 'replace'))
                    except UnicodeEncodeError:
                        logging.error('UnicodeEncodeError again with utf8-replace at file {} at line {}'.format(file_name, count))
                        if default_line is not None:
                            file.write(json.dumps(default_line, ensure_ascii=False))
                            logging.error('Write default {} at file {} at line {}'.format(default_line, file_name, count))
                        else:
                            logging.error('Ignore file {} at line {}'.format(file_name, count))
                            count -= 1
                file.write('\n')
                count += 1

    @staticmethod
    def read_csv_file_as_df(file_name) -> pandas.DataFrame:
        df = pandas.read_csv(file_name)
        return df

    @staticmethod
    def df_to_list_dict(df: pandas.DataFrame) -> List[dict]:
        dic = df.T.to_dict()
        lis = []
        for k, v in dic.items():
            lis.append(v)
        return lis

    @staticmethod
    def read_csv_file_as_list_dict(file_name, sep=',') -> List[dict]:
        df = pandas.read_csv(file_name, sep=sep)
        lis = UtilData.df_to_list_dict(df)
        return lis

    @staticmethod
    def save_pickle(obj, path):
        logging.info("saving pickle to: {}".format(path))
        with open(path, 'wb') as file:
            pickle.dump(obj, file)

    @staticmethod
    def load_pickle(path):
        logging.info("loading pickle from: {}".format(path))
        with open(path, 'rb') as file:
            obj = pickle.load(file)
        return obj

    @staticmethod
    def get_all_sub_dir_files_path(path) -> list:
        files = os.listdir(path)
        files = [path / x for x in files]
        return files

    @staticmethod
    def write_lines_text_file_error_catching(file_name, data, verbose=True):
        if verbose:
            logging.info("writing lines text to: {}".format(file_name))
        with open(file_name, 'w', encoding='utf-8') as file:
            for line in data:
                try:
                    file.write(line)
                    file.write('\n')
                except Exception as e:
                    logging.error('error with file {}: {}'.format(file_name, e))
                    pass

    @staticmethod
    def decimal_to_rgb(decimal):
        hexadecimal = "{:06x}".format(decimal)
        r = int(hexadecimal[0:2], 16)
        g = int(hexadecimal[2:4], 16)
        b = int(hexadecimal[4:6], 16)
        return r, g, b
