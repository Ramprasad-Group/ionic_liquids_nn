#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from rdkit import Chem
from rdkit.Chem import Fragments
from rdkit.Chem import rdMolDescriptors
mqn = rdMolDescriptors.MQNs_

import inspect
from selenium import webdriver
import re
import time
import xlwt
from xlwt import Workbook
import xlrd
from xlutils.copy import copy
import numpy as np
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import Select
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

book = xlrd.open_workbook("C:[INSERT TARGET FILE LOCATION AND NAME].xlsx")
worksheet = book.sheet_by_index(0)

#new_wb = copy(book)
#new_s = new_wb.get_sheet(0)

mol_smiles_list = []
comp_list = []
fragment_list = []

for i in range(4341): #change range to the number of rows in sheet

    cell_val = worksheet.cell(i,2).value #checks the second column of every row

    smile = cell_val
    mol_str = Chem.MolFromSmiles(str(smile))
    mol_smiles_list.append(mol_str)

        
#slecting fragments for ionic liquids

functions = dir(Chem.Fragments)
del functions[0:15]
del functions[86]
del functions[85]
functions_1 = [e[3:] for e in functions]

#slecting molecular descriptors for ionic liquids

functions_2 = dir(Chem.rdMolDescriptors)
del functions_2[0:5]
del functions_2[10:21]
del functions_2[14:18]
del functions_2[36:]
functions_2 = [e[4:] for e in functions_2]

#adding MQNs to the fingerprint

mqn = rdMolDescriptors.MQNs_
mqn_len = list(range(1,43))
mq_len = ["MQN_" + str(mq) for mq in mqn_len]
    

all_func = functions_1 + functions_2 + mq_len


df = pd.DataFrame(fragment_list, columns = [all_func], index = [comp_list])

print (len(mol_smiles_list))

r = 0
count = 0

#creating the fingerprint for each SMILE 

for item in mol_smiles_list:
    try:

        fp = Chem.Fragments.fr_Al_COO(item)
        fragment_list.append(fp)
        fp = Chem.Fragments.fr_Al_OH(item)
        fragment_list.append(fp)
        fp = Chem.Fragments.fr_Al_OH_noTert(item)
        fragment_list.append(fp)
        fp = Chem.Fragments.fr_ArN(item)
        fragment_list.append(fp)
        fp = Chem.Fragments.fr_Ar_COO(item)
        fragment_list.append(fp)
        fp = Chem.Fragments.fr_Ar_N(item)
        fragment_list.append(fp)
        fp = Chem.Fragments.fr_Ar_NH(item)
        fragment_list.append(fp)
        fp = Chem.Fragments.fr_Ar_OH(item)
        fragment_list.append(fp)
        fp = Chem.Fragments.fr_COO(item)
        fragment_list.append(fp)
        fp = Chem.Fragments.fr_COO2(item)
        fragment_list.append(fp)
        fp = Chem.Fragments.fr_C_O(item)
        fragment_list.append(fp)
        fp = Chem.Fragments.fr_C_O_noCOO(item)
        fragment_list.append(fp)
        fp = Chem.Fragments.fr_C_S(item)
        fragment_list.append(fp)
        fp = Chem.Fragments.fr_HOCCN(item)
        fragment_list.append(fp)
        fp = Chem.Fragments.fr_Imine(item)
        fragment_list.append(fp)
        fp = Chem.Fragments.fr_NH0(item)
        fragment_list.append(fp)
        fp = Chem.Fragments.fr_NH1(item)
        fragment_list.append(fp)
        fp = Chem.Fragments.fr_NH2(item)
        fragment_list.append(fp)
        fp = Chem.Fragments.fr_N_O(item)
        fragment_list.append(fp)
        fp = Chem.Fragments.fr_Ndealkylation1(item)
        fragment_list.append(fp)
        fp = Chem.Fragments.fr_Ndealkylation2(item)
        fragment_list.append(fp)
        fp = Chem.Fragments.fr_Nhpyrrole(item)
        fragment_list.append(fp)
        fp = Chem.Fragments.fr_SH(item)
        fragment_list.append(fp)
        fp = Chem.Fragments.fr_aldehyde(item)
        fragment_list.append(fp)
        fp = Chem.Fragments.fr_alkyl_carbamate(item)
        fragment_list.append(fp)
        fp = Chem.Fragments.fr_alkyl_halide(item)
        fragment_list.append(fp)
        fp = Chem.Fragments.fr_allylic_oxid(item)
        fragment_list.append(fp)
        fp = Chem.Fragments.fr_amide(item)
        fragment_list.append(fp)
        fp = Chem.Fragments.fr_amidine(item)
        fragment_list.append(fp)
        fp = Chem.Fragments.fr_aniline(item)
        fragment_list.append(fp)
        fp = Chem.Fragments.fr_aryl_methyl(item)
        fragment_list.append(fp)
        fp = Chem.Fragments.fr_azide(item)
        fragment_list.append(fp)
        fp = Chem.Fragments.fr_azo(item)
        fragment_list.append(fp)
        fp = Chem.Fragments.fr_barbitur(item)
        fragment_list.append(fp)
        fp = Chem.Fragments.fr_benzene(item)
        fragment_list.append(fp)
        fp = Chem.Fragments.fr_benzodiazepine(item)
        fragment_list.append(fp)
        fp = Chem.Fragments.fr_bicyclic(item)
        fragment_list.append(fp)
        fp = Chem.Fragments.fr_diazo(item)
        fragment_list.append(fp)
        fp = Chem.Fragments.fr_dihydropyridine(item)
        fragment_list.append(fp)
        fp = Chem.Fragments.fr_epoxide(item)
        fragment_list.append(fp)
        fp = Chem.Fragments.fr_ester(item)
        fragment_list.append(fp)
        fp = Chem.Fragments.fr_ether(item)
        fragment_list.append(fp)
        fp = Chem.Fragments.fr_furan(item)
        fragment_list.append(fp)
        fp = Chem.Fragments.fr_guanido(item)
        fragment_list.append(fp)
        fp = Chem.Fragments.fr_halogen(item)
        fragment_list.append(fp)
        fp = Chem.Fragments.fr_hdrzine(item)
        fragment_list.append(fp)
        fp = Chem.Fragments.fr_hdrzone(item)
        fragment_list.append(fp)
        fp = Chem.Fragments.fr_imidazole(item)
        fragment_list.append(fp)
        fp = Chem.Fragments.fr_imide(item)
        fragment_list.append(fp)
        fp = Chem.Fragments.fr_isocyan(item)
        fragment_list.append(fp)
        fp = Chem.Fragments.fr_isothiocyan(item)
        fragment_list.append(fp)
        fp = Chem.Fragments.fr_ketone(item)
        fragment_list.append(fp)
        fp = Chem.Fragments.fr_ketone_Topliss(item)
        fragment_list.append(fp)
        fp = Chem.Fragments.fr_lactam(item)
        fragment_list.append(fp)
        fp = Chem.Fragments.fr_lactone(item)
        fragment_list.append(fp)
        fp = Chem.Fragments.fr_methoxy(item)
        fragment_list.append(fp)
        fp = Chem.Fragments.fr_morpholine(item)
        fragment_list.append(fp)
        fp = Chem.Fragments.fr_nitrile(item)
        fragment_list.append(fp)
        fp = Chem.Fragments.fr_nitro(item)
        fragment_list.append(fp)
        fp = Chem.Fragments.fr_nitro_arom(item)
        fragment_list.append(fp)
        fp = Chem.Fragments.fr_nitro_arom_nonortho(item)
        fragment_list.append(fp)
        fp = Chem.Fragments.fr_nitroso(item)
        fragment_list.append(fp)
        fp = Chem.Fragments.fr_oxazole(item)
        fragment_list.append(fp)
        fp = Chem.Fragments.fr_oxime(item)
        fragment_list.append(fp)
        fp = Chem.Fragments.fr_para_hydroxylation(item)
        fragment_list.append(fp)
        fp = Chem.Fragments.fr_phenol(item)
        fragment_list.append(fp)
        fp = Chem.Fragments.fr_phenol_noOrthoHbond(item)
        fragment_list.append(fp)
        fp = Chem.Fragments.fr_phos_acid(item)
        fragment_list.append(fp)
        fp = Chem.Fragments.fr_phos_ester(item)
        fragment_list.append(fp)
        fp = Chem.Fragments.fr_piperdine(item)
        fragment_list.append(fp)
        fp = Chem.Fragments.fr_piperzine(item)
        fragment_list.append(fp)
        fp = Chem.Fragments.fr_priamide(item)
        fragment_list.append(fp)
        fp = Chem.Fragments.fr_prisulfonamd(item)
        fragment_list.append(fp)
        fp = Chem.Fragments.fr_pyridine(item)
        fragment_list.append(fp)
        fp = Chem.Fragments.fr_quatN(item)
        fragment_list.append(fp)
        fp = Chem.Fragments.fr_sulfide(item)
        fragment_list.append(fp)
        fp = Chem.Fragments.fr_sulfonamd(item)
        fragment_list.append(fp)
        fp = Chem.Fragments.fr_sulfone(item)
        fragment_list.append(fp)
        fp = Chem.Fragments.fr_term_acetylene(item)
        fragment_list.append(fp)
        fp = Chem.Fragments.fr_tetrazole(item)
        fragment_list.append(fp)
        fp = Chem.Fragments.fr_thiazole(item)
        fragment_list.append(fp)
        fp = Chem.Fragments.fr_thiocyan(item)
        fragment_list.append(fp)
        fp = Chem.Fragments.fr_thiophene(item)
        fragment_list.append(fp)
        fp = Chem.Fragments.fr_unbrch_alkane(item)
        fragment_list.append(fp)
        fp = Chem.Fragments.fr_urea(item)  
        fragment_list.append(fp)
        fp_2 = Chem.rdMolDescriptors.CalcChi0n(item)
        fragment_list.append(fp_2)
        fp_2 = Chem.rdMolDescriptors.CalcChi0v(item)
        fragment_list.append(fp_2)
        fp_2 = Chem.rdMolDescriptors.CalcChi1n(item)
        fragment_list.append(fp_2)
        fp_2 = Chem.rdMolDescriptors.CalcChi1v(item)
        fragment_list.append(fp_2)
        fp_2 = Chem.rdMolDescriptors.CalcChi2n(item)
        fragment_list.append(fp_2)
        fp_2 = Chem.rdMolDescriptors.CalcChi2v(item)
        fragment_list.append(fp_2)
        fp_2 = Chem.rdMolDescriptors.CalcChi3n(item)
        fragment_list.append(fp_2)
        fp_2 = Chem.rdMolDescriptors.CalcChi3v(item)
        fragment_list.append(fp_2)
        fp_2 = Chem.rdMolDescriptors.CalcChi4n(item)
        fragment_list.append(fp_2)
        fp_2 = Chem.rdMolDescriptors.CalcChi4v(item)
        fragment_list.append(fp_2)
        fp_2 = Chem.rdMolDescriptors.CalcKappa1(item)
        fragment_list.append(fp_2)
        fp_2 = Chem.rdMolDescriptors.CalcKappa2(item)
        fragment_list.append(fp_2)
        fp_2 = Chem.rdMolDescriptors.CalcKappa3(item)
        fragment_list.append(fp_2)
        fp_2 = Chem.rdMolDescriptors.CalcLabuteASA(item)
        fragment_list.append(fp_2)
        fp_2 = Chem.rdMolDescriptors.CalcNumAliphaticCarbocycles(item)
        fragment_list.append(fp_2)
        fp_2 = Chem.rdMolDescriptors.CalcNumAliphaticHeterocycles(item)
        fragment_list.append(fp_2)
        fp_2 = Chem.rdMolDescriptors.CalcNumAliphaticRings(item)
        fragment_list.append(fp_2)
        fp_2 = Chem.rdMolDescriptors.CalcNumAmideBonds(item)
        fragment_list.append(fp_2)
        fp_2 = Chem.rdMolDescriptors.CalcNumAromaticCarbocycles(item)
        fragment_list.append(fp_2)
        fp_2 = Chem.rdMolDescriptors.CalcNumAromaticHeterocycles(item)
        fragment_list.append(fp_2)
        fp_2 = Chem.rdMolDescriptors.CalcNumAromaticRings(item)
        fragment_list.append(fp_2)
        fp_2 = Chem.rdMolDescriptors.CalcNumAtomStereoCenters(item)
        fragment_list.append(fp_2)
        fp_2 = Chem.rdMolDescriptors.CalcNumBridgeheadAtoms(item)
        fragment_list.append(fp_2)
        fp_2 = Chem.rdMolDescriptors.CalcNumHBA(item)
        fragment_list.append(fp_2)
        fp_2 = Chem.rdMolDescriptors.CalcNumHBD(item)
        fragment_list.append(fp_2)
        fp_2 = Chem.rdMolDescriptors.CalcNumHeteroatoms(item)
        fragment_list.append(fp_2)
        fp_2 = Chem.rdMolDescriptors.CalcNumHeterocycles(item)
        fragment_list.append(fp_2)
        fp_2 = Chem.rdMolDescriptors.CalcNumLipinskiHBA(item)
        fragment_list.append(fp_2)
        fp_2 = Chem.rdMolDescriptors.CalcNumLipinskiHBD(item)
        fragment_list.append(fp_2)
        fp_2 = Chem.rdMolDescriptors.CalcNumRings(item)
        fragment_list.append(fp_2)
        fp_2 = Chem.rdMolDescriptors.CalcNumRotatableBonds(item)
        fragment_list.append(fp_2)
        fp_2 = Chem.rdMolDescriptors.CalcNumSaturatedCarbocycles(item)
        fragment_list.append(fp_2)
        fp_2 = Chem.rdMolDescriptors.CalcNumSaturatedHeterocycles(item)
        fragment_list.append(fp_2)
        fp_2 = Chem.rdMolDescriptors.CalcNumSaturatedRings(item)
        fragment_list.append(fp_2)
        fp_2 = Chem.rdMolDescriptors.CalcNumSpiroAtoms(item)
        fragment_list.append(fp_2)
        fp_2 = Chem.rdMolDescriptors.CalcNumUnspecifiedAtomStereoCenters(item)
        fragment_list.append(fp_2)
        

        mqn_list = rdMolDescriptors.MQNs_(item)

        
        for t in mqn_list:
            fragment_list.append(t)
            
        df_length = len(df)
        df.loc[df_length] = fragment_list
        fragment_list = []
        
        count+= 1

    except:
        print ('ERROR', count)
        count+=1


#Transferring data from original spreadsheet into new spreadsheet with fingerprint
        
df_old = pd.read_csv("C:[INSERT TARGET FILE LOCATION AND NAME].csv", encoding = "ISO-8859-1")
temp = df_old.iloc[:,0]
ec = df_old.iloc[:,1]
smiles = df_old.iloc[:,-1]
length = (len(df.iloc[1]))

df['SMILE'] = smiles
df['Electrical Conductivity, S/m Liquid'] = ec
df['Temperature, K'] = temp


df.to_csv(r'C:[INSERT TARGET FILE LOCATION AND NAME].csv', index = True, header = True)
     

