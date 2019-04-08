import matplotlib.pyplot as plt
import os
from openpyxl import Workbook, load_workbook
import sys


def check_xml():
    filepath = './compare/results.xlsx'
    wb=load_workbook(filepath)
    try:
        wb.save(filepath)
    except PermissionError as error:
        print(error)
        print('Le fichier excel est déjà ouvert ! Merci de le fermer avant le lancement d\'un script.')
        sys.exit()


def check_overwrite(model):
    choice = 'null'
    weights = os.listdir('./weights/')
    new_file = model + '_actor.h5f'
    for weight in weights:
        if(new_file == weight):
            while(choice != 'y' and choice != 'n'):
                choice = input('Un model porte déjà ce nom, voulez-vous écrire par dessus ? (y/n)  ')
            if(choice == 'y'):
                pass
            elif(choice == 'n'):
                sys.exit()