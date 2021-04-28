import re
from parsy import regex, string, seq

def parse_xoopic_input(filename):
    '''
    Parses and evaluates the contents of XOOPIC input file.

    Returns:
        Dictionary corresponding to "Region" block in input file.

    Example:
        Get the time-step and the charge of species 0 and 1:

            config = parse_xoopic_input(filename)
            dt = config['Control'][0]['dt']
            q0 = config['Species'][0]['q']
            q1 = config['Species'][1]['q']

            from pprint import PrettyPrinter
            PrettyPrinter().pprint(config)
    '''
    contents = XoopicParser()(filename)
    evaluated_contents = eval_xoopic_region(contents)
    return evaluated_contents

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
    """
    Takes in a parsed xoopic input file, and evaluates all expression in the
    "Region" block.
    """

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
