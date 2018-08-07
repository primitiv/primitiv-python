#!/usr/bin/env python3

import os.path
import re
import sys

from argparse import ArgumentParser
from subprocess import Popen, PIPE


def get_function_name(x):
    funcname_end = x.index("(")
    funcname_start = funcname_end
    template_depth = 0
    while not (template_depth == 0 and x[funcname_start] == " "):
        if x[funcname_start] == ">":
            template_depth += 1
        if x[funcname_start] == "<":
            template_depth -= 1
        funcname_start -= 1
        if funcname_start == -1:
            break
    return x[funcname_start + 1:funcname_end]


def main():
    parser = ArgumentParser(description="Check binding coverage and list unimplemented functions")
    parser.add_argument("core_library", help="path to the core library")
    parser.add_argument("binding_library", help="path to the binding library (built with --no-build-core-library)")
    parser.add_argument("--namespace", "-n", help="namespace", default="")
    args = parser.parse_args()

    nm_re = re.compile("([0-9a-f ]{16}) (.) (.+)")
    nm_core = Popen(["nm", "--demangle", args.core_library], stdout=PIPE)
    nm_binding = Popen(["nm", "--demangle", "--undefined-only", args.binding_library], stdout=PIPE)
    core_symbols = [x for x in nm_core.stdout.read().decode("utf-8").split("\n")]
    binding_symbols = [x for x in nm_binding.stdout.read().decode("utf-8").split("\n")]
    core_functions = set()
    binding_functions = set()
    function_head = args.namespace + "::" if args.namespace else ""
    for s in core_symbols:
        m = nm_re.match(s)
        if m:
            symbol_name = m.group(3)
            if "(" in symbol_name:
                if get_function_name(symbol_name).startswith(function_head):
                    core_functions.add(symbol_name)
    for s in binding_symbols:
        m = nm_re.match(s)
        if m:
            symbol_name = m.group(3)
            if "(" in symbol_name:
                if get_function_name(symbol_name).startswith(function_head):
                    binding_functions.add(symbol_name)

    functions_total = len(core_functions)
    binding_total = len(binding_functions)

    if functions_total == 0:
        print("No function found.", file=sys.stderr)
        return 2

    diff_core_binding = core_functions - binding_functions
    diff_binding_core = binding_functions - core_functions

    l = [(-1, x) for x in diff_core_binding] + [(1, x) for x in diff_binding_core]
    l.sort(key=lambda x: (x[1], x[0]))

    print("Difference:")
    for x in l:
        if x[0] == -1:
            print("\033[31m- %s\033[0m" % x[1])
        else:
            print("\033[32m+ %s\033[0m" % x[1])
    print()
    functions_total = len(core_functions)
    binding_total = len(binding_functions.intersection(core_functions))
    print("Coverage: %.2f%% (%d/%d)" % (binding_total / functions_total * 100, binding_total, functions_total))

    if core_functions == binding_functions:
        return 0
    else:
        return 1


if __name__ == "__main__":
    sys.exit(main())
