import argparse
import sys

import global_time_series

def main():

    # Parser
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        usage="""zppy-interfaces <interface> [<args>]

Available interfaces:
  global-time-series   Generate Global Time Series plots

For help with a specific interface
  zppy-interfaces interface --help
""")
    parser.add_argument("interface", help="interface to use (e.g., global_time_series)")
    # parse_args defaults to [1:] for args, but you need to
    # exclude the rest of the args too, or validation will fail
    args: argparse.Namespace = parser.parse_args(sys.argv[1:2])

    if args.interface == "global_time_series":
        global_time_series.global_time_series()
    else:
        print("Unrecognized interface")
        parser.print_help()
        sys.exit(1)