#!/usr/bin/env python3

import re
import sys
from parsy import generate, regex, string, seq, letter, digit, char_from
from collections import namedtuple

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

def parse_xoopic(filename):

    with open(filename) as file:
        code = file.read()

    # Remove comments
    code = re.sub('//.*', '', code)

    # Remove leading and trailing whitespaces (incl. empty lines)
    code = re.sub(r'^\s*', '', code, flags=re.MULTILINE)
    code = re.sub(r'\s*$', '', code, flags=re.MULTILINE)

    spaces = regex(r'[ \t]*')
    whitespace = regex(r'\s*')
    horspaces = regex('r[ \t]*')

    newline = string('\n')
    lbrace  = whitespace << string('{') << whitespace
    rbrace  = whitespace << string('}') << whitespace
    equal   = string('=')
    # word    = regex('[a-zA-Z0-9\-.,*_:%]+')
    word    = regex('[^\s=}]+')
    # words   = regex('[a-zA-Z0-9\-._: ]*')
    words   = word + (spaces+word).many().concat()

    # whitespace = regex(r'\s*')
    # lexeme  = lambda p: whitespace >> p << whitespace
    # lbrace  = lexeme(string('{'))
    # rbrace  = lexeme(string('}'))
    # equal   = lexeme(string('='))
    # word    = lexeme((letter|digit|char_from('.-_%:')).at_least(1).concat())

    key_value_line = seq(spaces >> word << spaces << equal, spaces >> words << spaces)
    key_value_dict = key_value_line.sep_by(newline).map(dict)
    # key_value_dict = key_value_line.many().map(dict)
    # key_value_dict = key_value_line.map(namedtuple).many()
    # res = p.parse(' key =  value ')
    # print(res)

    block = seq(word << lbrace, key_value_dict << rbrace)
    comment_block = seq(word << lbrace, regex(r'[^}]*') << rbrace)
    # res = block.many().parse(s)
    region_block = seq(whitespace >> string('Region') << lbrace, block.many() << rbrace)
    variables_block = seq(whitespace >> string('Variables') << lbrace, regex(r'[^}]*') << rbrace)
    plasmadiag_block = seq(whitespace >> string('Plasma_Diagnostics') << lbrace, regex(r'[^}]*') << rbrace)
    xoopic_file = seq(plasmadiag_block, variables_block, region_block)
    # xoopic_file = (variables_block | region_block | plasmadiag_block).many()
    # res = region_block.parse(s)
    # res = xoopic_file.parse_partial(s)
    # for n, line in enumerate(code.split('\n')):
    #     print(n, line)
    res = xoopic_file.parse(code)

    for key in res:
        print(key[0])
    # print(res[1]['temperature'])

    return res




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
