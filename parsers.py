import re
from parsy import regex, string, seq
from copy import deepcopy
import numpy as np
from functools import reduce

class XoopicParser(object):
    """
    A bare bones parser for XOOPIC input files. Example of use:

        file_contents = XoopicParser()(filename)
        PrettyPrinter.pprint(file_contents)
    """

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

def eval_xoopic(xoopic_input):
    """
    Takes in a parsed xoopic input file, and evaluates all expression in the
    "Variables" and "Region" blocks.

    Example:
        Get the time-step and the charge of species 0 and 1:

        region, variables = parse_xoopic_input(filename)
        dt = region['Control'][0]['dt']
        q0 = region['Species'][0]['q']
        q1 = region['Species'][1]['q']

        from pprint import PrettyPrinter
        PrettyPrinter().pprint(region)
    """

    # Import sqrt and other mathematical function into Variables block
    code = xoopic_input['Variables']
    code = 'from math import *\n' + code

    # Execute as Python and store variables
    variables = {}
    exec(code, None, variables)

    # Evaluate each value in the region block using the variables
    region = xoopic_input['Region']
    for subblock in region.values():
        for parameters in subblock:
            for key in parameters:
                try:
                    parameters[key] = eval(parameters[key], None, variables)
                except:
                    pass

    return region, variables

def parse_xoopic_input(filename):
    """
    Convenience function for parsing XOOPIC files, using eval_xoopic and
    XoopicParser. Returns a dictionary.
    """
    contents = XoopicParser()(filename)
    region, variables = eval_xoopic(contents)

    d = {'Region': region, 'Variables': variables}

    grid = region['Grid'][0]
    dx = (grid['x1f']-grid['x1s'])/grid['J']
    dy = (grid['x2f']-grid['x2s'])/grid['K']
    d['dx'] = np.array([dx, dy])

    d['dt'] = region['Control'][0]['dt']
    d['plasma'] = infer_plasma_parameters(region, variables)

    return d

def dict_of_list(list_of_pairs):
    """
    Creates a dictionary from a list of (key,value)-pairs.
    However, to allow multiple identical keys, each entry
    will be a list of values belonging to that key.
    """
    dict_of_list = {}
    for key, value in list_of_pairs:
        if key not in dict_of_list:
            dict_of_list[key] = []
        dict_of_list[key].append(value)
    return dict_of_list


def infer_plasma_parameters(config, variables, langmuir_format=True):
    species = {}
    for s in config['Species']:
        name = s['name']
        species[name] = {'q': s['q'], 'm': s['m']}

    plasma = {}
    for blockname in ['Load', 'EmitPort']:
        if blockname in config:
            for s in config[blockname]:

                name = s['speciesName']
                if name not in plasma:
                    plasma[name] = []

                q = species[name]['q']
                m = species[name]['m']
                vth = s['temperature']
                u = np.array([s['v1drift'], s['v2drift'], s['v3drift']])

                try:
                    n = s['density']
                except:
                    try:
                        varname = 'nI' if q>0 else 'nE'
                        n = variables[varname]
                        print('Density not available in {} block. '
                              'Guessing "{}" is the density for "{}".'
                              .format(blockname, varname, name))
                    except:
                        try:
                            varname = 'Ni' if q>0 else 'Ne'
                            n = variables[varname]
                            print('Density not available in {} block. '
                                  'Guessing "{}" is the density for "{}".'
                                  .format(blockname, varname, name))
                        except:
                            n = None

                d = {'q': q, 'm': m, 'n': n, 'vth': vth, 'u': u}
                plasma[name].append(d)

    if langmuir_format:
        plasma = convert_plasma_to_langmuir(plasma)

    return plasma

def convert_plasma_to_langmuir(plasma):
    from langmuir import Species
    plasma = deepcopy(plasma)
    for p in plasma:
        for i, q in enumerate(plasma[p]):
            u=q.pop('u')
            plasma[p][i] = Species(**q)
            plasma[p][i].u = u
    # plasma = reduce(lambda x, acc: x+acc, plasma.values())
    # plasma.sort(key=lambda x: x.q)
    return plasma
