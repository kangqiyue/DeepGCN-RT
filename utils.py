import torch

# device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def count_no_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if not p.requires_grad)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())



if __name__ == "__main__":
    '''split excel sheet'''
    import pandas as pd
    import os

    path = "D:\DEEPGNN_RT\dataset\RIKEN"
    os.chdir(path)
    print(os.getcwd())

    xl = pd.ExcelFile('ac9b05765_si_001 (1).xlsx')

    for sheet in xl.sheet_names:
        df = pd.read_excel(xl, sheet_name=sheet)
        df.columns = df.iloc[1, :]
        df = df.iloc[2:, :]
        df["rt"] = df['Experimental Retention Time'] *60
        df["smiles"] = df["SMILES"]

        name = sheet.split("_")[-2] + "_" + sheet.split("_")[-1]

        df.to_excel(f"{name}.xlsx", index=False)





