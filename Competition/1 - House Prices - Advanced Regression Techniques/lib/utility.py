import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt

from rich import print
from rich.columns import Columns
from rich.console import Console
from rich.panel import Panel

from textwrap import fill

################################################
# Print / Plot
################################################
def color(text, color):
    return f"[{color}]{text}[/]"

def cyan(text):
    return color(text, "bright_cyan")

def green(text):
    return color(text, "bright_green")

def red(text):
    return color(text, "bright_red")

def white(text):
    return color(text, "bright_white")

def yellow(text):
    return color(text, "bright_yellow")
