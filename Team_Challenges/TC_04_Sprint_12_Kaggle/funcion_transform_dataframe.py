import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
import urllib.request
import bootcampviztools as bt
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from xgboost import XGBRegressor
from catboost import CatBoostRegressor,Pool
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import funcion_transform_dataframe as tdf
import pandas as pd
import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from scipy.stats import randint, uniform
from sklearn.model_selection import cross_val_score
import re
''' 
No sabía aplicar todas estas tranformaciones con un pipeline, porque habíam mucha creación de columnas en mi procesado. 
No sabía si iba a afectar al test o no o si se iba a replicar de la misma forma.
Esta función contiene funciones que añade o limpia columnas, y después con el pipeline se realizan las transformaciones (encoder, logs...etc).
De modo que todos los cambios manuales que hice antes de encontrarme este problema las he metido en esta función. Cruzamos los dedos para que funcione.
'''


def transformar_dataset(df):
    df = df.copy()

    # RAM (qiutar GB)
    df["Ram"] = df["Ram"].str.replace("GB", "", regex=False)

    
    # weight (quitar kg)
    df["Weight"] = df["Weight"].str.replace("kg", "", regex=False).astype(float)


    # CPU marca y familia (coger Intel o AMD y la familia, sino se queda en otros, se hacen doa nuevas cols)
    df["Cpu_marca"] = df["Cpu"].str.split().str[0]
    df["Cpu_familia"] = df["Cpu"].str.split().str[1]

    
    # CPU serie (un poco random porque CPU tiene muchos datos con diferente estructura, pero puede valer con estas normas. Si vienen datos muy diferentes petará)
    def extraer_serie(cpu):
        cpu = cpu.lower()
        if "core i7" in cpu: return "i7"
        if "core i5" in cpu: return "i5"
        if "core i3" in cpu: return "i3"
        if "core m" in cpu: return "Core M"
        if "celeron" in cpu: return "Celeron"
        if "pentium" in cpu: return "Pentium"
        if "atom" in cpu: return "Atom"
        if "a6-" in cpu: return "A6"
        if "a9-" in cpu: return "A9"
        if "a10-" in cpu: return "A10"
        if "e2-" in cpu: return "E2"
        return "Otro"

    df["Cpu_serie"] = df["Cpu"].apply(extraer_serie)

    
    # CPU GHz (cojo la frecencia de la ultima posiciony s e hace una nueva col, quita GHz)
    df["Cpu_GHz"] = df["Cpu"].str.split().str[-1]
    df["Cpu_GHz"] = df["Cpu_GHz"].str.replace("GHz", "", regex=False)
    df["Cpu_GHz"] = pd.to_numeric(df["Cpu_GHz"], errors="coerce")

    
    # CPU código (esto es un poco frankenstein, pero a ver si funciona, saco el codigo por un lado y si hay muchos revisar se vuelve a mirar)
    df["Cpu_codigo"] = df["Cpu"].str.split().str[-2]

    def extraer_codigo(cpu):
        cpu = cpu.replace("‑", "-").replace("–", "-").replace("—", "-")
        cpu = cpu.upper().split()

        for p in cpu:
            if "X5-Z" in p:
                return p
        for p in cpu:
            if (p[:-1].isdigit() and p[-1] in "UQKHY") or \
               (len(p) > 2 and p[:-2].isdigit() and p[-2:] in ["HQ", "HK"]):
                return p
        for p in cpu:
            if p.startswith("N") and p[1:].isdigit():
                return p
        for p in cpu:
            if p.startswith(("A6-", "A8-", "A9-", "A10-", "A12-")):
                return p
        for p in cpu:
            if p.startswith("E2-"):
                return p
        for p in cpu:
            if p.isdigit() and len(p) == 4 and p.startswith("1"):
                return p

        return "REVISAR"

    df["Cpu_codigo_bien"] = df["Cpu"].apply(extraer_codigo)

    def coger_codigo(cpu): #todavia hay mucha suciedad
        partes = cpu.split()
        if len(partes) >= 3:
            if len(partes[-2]) > 3:
                return partes[-2]
            else:
                return partes[-3]
        elif len(partes) == 2:
            return partes[0]
        else:
            return partes[0]

    mask_revisar = df["Cpu_codigo_bien"] == "REVISAR"
    df.loc[mask_revisar, "Cpu_codigo_bien"] = df.loc[mask_revisar, "Cpu"].apply(coger_codigo)

    codigos_malos = ["Core","A9-SERIES","A6-SERIES","A12-SERIES","A10-SERIES","A8-SERIES","M","E2-9000"]
    df.loc[df["Cpu_codigo_bien"].isin(codigos_malos), "Cpu_codigo_bien"] = df["Cpu"].str.split().str[-2]

    df.loc[df["Cpu_codigo_bien"].isin(["i5","i7","M","m3"]), "Cpu_codigo_bien"] = "UNK"

    
    # CPU generación (la que mas me ha costado y he tenido que informarme de que tenia q poner aqui)
    def generacion_cpu(codigo):
        codigo = str(codigo)
        if codigo[0].isdigit() and len(codigo) >= 4:
            return int(codigo[0])
        return "REVISAR"

    df["Cpu_generacion"] = df["Cpu_codigo"].apply(generacion_cpu)

    def obtener_generacion(cpu, modelo):
        cpu = cpu.upper()
        modelo = modelo.upper()
        if any(pref in modelo for pref in ["I3","I5","I7"]) and any(c.isdigit() for c in modelo):
            for token in modelo.split("-"):
                if token[0].isdigit():
                    return int(token[0])
        if modelo in ["I3","I5","I7"]:
            return "UNK"
        if any(x in modelo for x in ["M3","M7","M-"]):
            if "6Y" in modelo: return 6
            if "7Y" in modelo: return 7
            return "0"
        return 0

    df.loc[df["Cpu_generacion"] == "REVISAR", "Cpu_generacion"] = df.apply(
        lambda row: obtener_generacion(row["Cpu"], row["Cpu_codigo"]), axis=1
    )
    df["Cpu_generacion"] = df["Cpu_generacion"].replace(["UNK", "REVISAR"], np.nan)
    df["Cpu_generacion"] = pd.to_numeric(df["Cpu_generacion"], errors="coerce")


    
    # CPU sufijo (se saca a partir de codigo, sería la ultima letra del codigo (o donde la lleve si la derecta) que da mucha información interesamte)
    def cpu_sufijo(codigo):
        sufijo = re.findall(r'([A-Za-z]+)$', codigo)
        if sufijo:
            return sufijo[0]
        mid = re.findall(r'\d([A-Za-z]{1,2})\d', codigo)
        if mid:
            return mid[0]
        prefijo = re.findall(r'^([A-Za-z]+)', codigo)
        if prefijo:
            return prefijo[0]
        return "REVISAR"

    df["Cpu_letra"] = df["Cpu_codigo"].apply(cpu_sufijo)

    validos = ["U","HQ","HK","N","P","Y","M","E","V"]

    def limpiar_sufijo(s):
        s = str(s).strip()
        if s in validos:
            return s
        return "REVISAR"

    df["Cpu_letra_limpio"] = df["Cpu_letra"].apply(limpiar_sufijo)

    def clasificar_revisar(cpu):
        cpu = cpu.upper()
        if "RYZEN" in cpu and (" 1" in cpu or "-1" in cpu):
            return "DESKTOP"
        if any(x in cpu for x in ["A4","A6","A8","A9"]):
            return "A-SERIES"
        if "E1" in cpu or "E2" in cpu or "E-" in cpu or "E " in cpu:
            return "E-SERIES"
        if "ATOM" in cpu:
            return "ATOM"
        return "REVISAR"

    mask_revisar = df["Cpu_letra_limpio"] == "REVISAR"
    df.loc[mask_revisar, "Cpu_letra_limpio"] = df.loc[mask_revisar, "Cpu"].apply(clasificar_revisar)

    df["Cpu_letra_limpio"].replace("REVISAR","Xeon", inplace=True)
    df["Cpu_letra_limpio"].replace("V","Xeon", inplace=True)

    
    # gPU (de aqui tambien saco varias columnas, marca, famila, modelo y un binario de si es integrada o no. mucha documentación previa y se ajusta demasiadoa los datos actuales... puede ser que falle con nuevos)
    # .1
    df["Gpu_marca"] = df["Gpu"].str.split().str[0]

    def obtener_familia(gpu):
        gpu = gpu.strip()
        if gpu.startswith("Intel"):
            if "Iris Plus" in gpu: return "Iris Plus"
            if "Iris Pro" in gpu: return "Iris Pro"
            if "Iris" in gpu: return "Iris"
            if "UHD" in gpu: return "UHD"
            if "HD" in gpu: return "HD"
            if "Graphics" in gpu: return "Graphics"
            return "Intel_Otro"
        if gpu.startswith("Nvidia"):
            if "GeForce" in gpu: return "GeForce"
            if "Quadro" in gpu: return "Quadro"
            return "Nvidia_Otro"
        if gpu.startswith("AMD"):
            if "Radeon" in gpu: return "Radeon"
            if "FirePro" in gpu: return "FirePro"
            if "R" in gpu and "Graphics" in gpu: return "R-Series"
            return "AMD_Otro"
        return "OTRO"
    #.2
    df["Gpu_familia"] = df["Gpu"].apply(obtener_familia)

    def obtener_gpu_modelo(gpu):
        gpu = gpu.strip()
        m = re.search(r"(GTX|GT|MX|RX|R\d|M|W)?\s*([0-9]{3,4}[A-Z]*|[A-Z]?[0-9]{3,4}[A-Z]*)", gpu)
        if m:
            prefijo = m.group(1) or ""
            numero = m.group(2)
            return (prefijo + numero).strip()
        m = re.search(r"(\d{3,4})$", gpu)
        if m:
            return m.group(1)
        return "UNK"
    #.3
    df["Gpu_modelo"] = df["Gpu"].apply(obtener_gpu_modelo)
    #.4
    gpu_dedicadas = ["GTX","GT ","MX","Quadro","RX","FirePro","R5 M","R7 M","R9 M","R5M","R7M","R9M","Radeon Pro"]
    df["Gpu_tipo"] = "INTEGRADA"
    for patron in gpu_dedicadas:
        df.loc[df["Gpu"].str.contains(patron, case=False, na=False), "Gpu_tipo"] = "DEDICADA"

    
    # Memory
    df["Memory_num"] = df["Memory"].str.upper().str.replace("TB","000", regex=False)
    df["Memory_num"] = df["Memory_num"].apply(lambda x: "".join([c if c.isdigit() else " " for c in x]))
    df["col1"] = df["Memory_num"].str.split().str[0].fillna(0).astype(int)
    df["col2"] = df["Memory_num"].str.split().str[1].fillna(0).astype(int)
    df["Memory_num"] = df["col1"] + df["col2"]

    df["Memory_tipo"] = "Desconocido"
    df.loc[df["Memory"].str.contains("SSD", case=False, na=False) & df["Memory"].str.contains("HDD", case=False, na=False), "Memory_tipo"] = "Hibrido"
    df.loc[df["Memory"].str.contains("SSD", case=False, na=False) & ~df["Memory"].str.contains("HDD", case=False, na=False), "Memory_tipo"] = "SSD"
    df.loc[df["Memory"].str.contains("HDD", case=False, na=False) & ~df["Memory"].str.contains("SSD", case=False, na=False), "Memory_tipo"] = "HDD"
    df.loc[df["Memory"].str.contains("Flash", case=False, na=False), "Memory_tipo"] = "Flash"
    df.loc[df["Memory"].str.contains("Hybrid", case=False, na=False), "Memory_tipo"] = "Hybrid"

    
    # Screen resolution (aqui separo el ancho del alto de la pantalla y cada una sera una col),  espues las clasifico y hago un PPI (que son pixels per inch, cuanto mayo el PPI, mas calidad, la operacion es la raiz cuadrada del ancho2 mas el alto2 dividido por las inches)
    df["Screen_res"] = df["ScreenResolution"].str.extract(r"(\d{3,4}x\d{3,4})")
    df[["Screen_ancho","Screen_alto"]] = df["Screen_res"].str.split("x", expand=True)

    
    # PPI
    df["Screen_ancho"] = pd.to_numeric(df["Screen_ancho"], errors="coerce")
    df["Screen_alto"] = pd.to_numeric(df["Screen_alto"], errors="coerce")
    df["Inches"] = pd.to_numeric(df["Inches"], errors="coerce")

    df["PPI"] = np.sqrt(df["Screen_ancho"]**2 + df["Screen_alto"]**2) / df["Inches"]
    df["PPI"] = df["PPI"].fillna(0)

    
    # Clasificación resolución
    Screen_res_4k = ["3840x2160"]
    Screen_res_qhd = ["2560x1440","2560x1600","2304x1440","2160x1440"]
    Screen_res_retina = ["3200x1800","2880x1800","2400x1600","2256x1504"]
    Screen_res_fullhd = ["1920x1080","1920x1200"]
    Screenres_hd = ["1600x900","1440x900","1366x768"]

    df["Screen_res_tipo"] = "Other"
    df.loc[df["Screen_res"].isin(Screen_res_4k), "Screen_res_tipo"] = "4K"
    df.loc[df["Screen_res"].isin(Screen_res_qhd), "Screen_res_tipo"] = "QHD"
    df.loc[df["Screen_res"].isin(Screen_res_retina), "Screen_res_tipo"] = "Retina"
    df.loc[df["Screen_res"].isin(Screen_res_fullhd), "Screen_res_tipo"] = "FullHD"
    df.loc[df["Screen_res"].isin(Screenres_hd), "Screen_res_tipo"] = "HD"

    # Panel (los IPS son mas caros, los tn mas baratos)
    df["Screen_res_panel"] = "Other"
    df.loc[df["ScreenResolution"].str.contains("IPS", case=False, na=False), "Screen_res_panel"] = "IPS"

    #Touchscreen (+ precio)
    df["Screen_TouchScreen"] = 0
    df.loc[df["ScreenResolution"].str.contains("Touch", case=False, na=False), "Screen_TouchScreen"] = 1

    
    #  OS reducido, menos cats
    df["OS_reducido"] = "Other"
    df.loc[df["OpSys"].str.contains("Windows", case=False, na=False), "OS_reducido"] = "Windows"
    df.loc[df["OpSys"].str.lower().str.contains("mac", na=False), "OS_reducido"] = "Mac"
    df.loc[df["OpSys"].str.contains("Linux", case=False, na=False), "OS_reducido"] = "Linux"

    
    # Tipos numéricos, forzar
    float_cols = ["Inches","Weight","Cpu_GHz","PPI"]
    int_cols = ["Ram","Screen_ancho","Screen_alto","Memory_num"]

    for col in float_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in int_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    
    #Convertir categóricas a string,, frorzar
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].astype(str)

    

    #if "laptop_ID" in df.columns:
    #    df = df.drop(columns=["laptop_ID"])


    return df