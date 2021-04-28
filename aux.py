#!/usr/bin/env python3

import re
import sys
from parsy import regex, string, seq
from collections import namedtuple
from pprint import PrettyPrinter

def parse_xoopic_input(filename):
    '''
    Takes the name of a XOOPIC input file and returns a dictionary with all the
    variables in the Variable-block. NB: This is a hacky but convenient solution.
    '''

    with open(filename) as file:
        code = file.read()

    # Extract contents of Variables{...}
    code = re.search('Variables\s*?{([\s\S]*?)}', code).groups()[0]

    code = re.sub('//.*', '', code) # Remove comments
    code = re.sub('\n[ \t]*', '\n', code) # Remove leading whitespace

    # Import sqrt and other mathematical function
    code = 'from math import *\n' + code

    # Execute as Python
    vars = dict()
    exec(code, None, vars)

    return vars

def parse_xoopic_block(filename, blockname):
    '''
    Takes the name of a XOOPIC input file and returns a dictionary with all the
    variables in the Variable-block. NB: This is a hacky but convenient solution.
    '''

    with open(filename) as file:
        code = file.read()

    # Extract contents of block
    code = re.search(blockname+'\s*?{([\s\S]*?)}', code).groups()[0]

    # Remove comments
    code = re.sub('//.*', '', code)

    # Remove leading and trailing whitespaces (incl. empty lines)
    code = re.sub(r'(^\s*|\s*$)', '', code, flags=re.MULTILINE)

    code = code.split('\n')

    d = dict()
    for line in code:
        print(line)
        key, value = line.split('=')
        key = key.strip()
        value = value.strip()
        d[key] = value

    print(d)

    # # Execute as Python
    # vars = dict()
    # exec(code, None, vars)

    return vars

def dict_of_list(list_of_pairs):
    '''
    Creates a dictionary from a list of (key,value)-pairs.
    However, to allow multiple identical keys, each entry
    will be a list of values belonging to that key.
    '''
    dict_of_list = {}
    for key, value in list_of_pairs:
        if key not in dict_of_list:
            dict_of_list[key] = []
        dict_of_list[key].append(value)
    return dict_of_list

class XoopicParser(object):
    '''
    A parser for XOOPIC input files. Example of use:

        file_contents = XoopicParser()(filename)
        PrettyPrinter.pprint(file_contents)
    '''

    def __init__(self):

        spaces = regex(r'[ \t]*')  # Excludes newline
        whitespace = regex(r'\s*') # Includes newline

        newline = string('\n')
        equal   = string('=')
        lbrace  = whitespace << string('{') << whitespace
        rbrace  = whitespace << string('}') << whitespace

        # These parsers don't terminate blocks
        word    = regex('[^\s=}]+')
        words   = word + (spaces+word).many().concat()
        characters = regex(r'[^}]*')

        key_value_line = seq(spaces >> word << spaces << equal, spaces >> words << spaces)
        key_value_lines = key_value_line.sep_by(newline).map(dict)

        def block(name, content):
            return seq(whitespace >> name, lbrace >> content << rbrace)

        key_value_block = block(word, key_value_lines)
        key_value_blocks = key_value_block.many().map(dict_of_list)
        region_block = block(string('Region'), key_value_blocks)
        other_block = block(word, characters)

        self.parser = (region_block | other_block).many().map(dict)

    def __call__(self, filename):

        with open(filename) as file:
            code = file.read()

        # Remove comment
        code = re.sub('//.*', '', code)

        # Remove leading and trailing whitespaces (incl. empty lines)
        code = re.sub(r'^\s*', '', code, flags=re.MULTILINE)
        code = re.sub(r'\s*$', '', code, flags=re.MULTILINE)

        # FIXME: parser should handle comments and whitespaces through
        # combinators. Not ad-hoc.

        return self.parser.parse(code)

def eval_xoopic_region(xoopic_input):

    # Import sqrt and other mathematical function into Variables block
    code = xoopic_input['Variables']
    code = 'from math import *\n' + code

    # Execute as Python and store variables
    variables = {}
    exec(code, None, variables)

    # Evaluate each value in the region block using the variables
    region_block = xoopic_input['Region']
    for subblock in region_block.values():
        for parameters in subblock:
            for key in parameters:
                try:
                    parameters[key] = eval(parameters[key], None, variables)
                except:
                    pass

    return region_block

def read_xoopic_input(filename):
    '''
    Parses and evaluates the contents of XOOPIC input file.
    '''
    contents = XoopicParser()(filename)
    evaluated_contents = eval_xoopic_region(contents)
    return evaluated_contents


# def parse_xoopic_input2(filename):

#     with open(filename) as file:
#         code = file.read()

#     # Remove comment
#     code = re.sub('//.*', '', code)

#     # Remove leading and trailing whitespaces (incl. empty lines)
#     code = re.sub(r'^\s*', '', code, flags=re.MULTILINE)
#     code = re.sub(r'\s*$', '', code, flags=re.MULTILINE)

#     # FIXME: parser should handle comments and whitespaces

#     return xoopic_parser().parse(code)


# class XoopicParser(object):

#     def __init__(self, filename):

#         with open(filename) as file:
#             code = file.read()

#         # Remove comments
#         code = re.sub('//.*', '', code)

#         # Remove leading and trailing whitespaces (incl. empty lines)
#         code = re.sub(r'^\s*', '', code, flags=re.MULTILINE)
#         code = re.sub(r'\s*$', '', code, flags=re.MULTILINE)

#         self._parse(code)
#         self._eval_variables()

#     def _parse(self, code):

#         self.contents = xoopic_file.parse(code)

#     def _eval_variables(self):

#         # Import sqrt and other mathematical function
#         code = self.contents['Variables']
#         code = 'from math import *\n' + code

#         # Execute as Python
#         self.variables = dict()
#         exec(code, None, self.variables)

#     def __call__(self, section, key, evaluate=True):
#         values = []
#         for block in self.contents['Region']:
#             if block[0] == section:
#                 value = block[1][key]
#                 if evaluate:
#                     try:
#                         value = eval(value, None, self.variables)
#                     except:
#                         pass
#                 values.append(value)
#         print(values)
#         return values


    # blockname='Region'

    # # Extract contents of block
    # code = re.search(blockname+'\s*?{([\s\S]*?)}', code).groups()[0]
    # print(code)

    # code = code.split('\n')

    # d = dict()
    # for line in code:
    #     print(line)
    #     key, value = line.split('=')
    #     key = key.strip()
    #     value = value.strip()
    #     d[key] = value

    # print(d)

    # # # Execute as Python
    # # vars = dict()
    # # exec(code, None, vars)

    # return vars
