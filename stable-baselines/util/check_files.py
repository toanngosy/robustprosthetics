import matplotlib.pyplot as plt
import os
from openpyxl import Workbook, load_workbook
import sys


def check_overwrite(model):
    choice = 'null'
    weights = os.listdir('./models/')
    new_file = model + '.pkl'
    for weight in weights:
        if(new_file == weight):
            while(choice != 'y' and choice != 'n'):
                choice = input('Un model porte déjà ce nom, voulez-vous écrire par dessus ? (y/n)  ')
            if(choice == 'y'):
                pass
            elif(choice == 'n'):
                sys.exit()
