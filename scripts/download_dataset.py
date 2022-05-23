import os
import zipfile

os.system("wget -N http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip")
os.system("wget -N http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip")

if not os.path.exists("div2k"):
    os.makedirs("div2k")

with zipfile.ZipFile("DIV2K_train_HR.zip", "r") as zip_ref:
    zip_ref.extractall("div2k")

with zipfile.ZipFile("DIV2K_valid_HR.zip", "r") as zip_ref:
    zip_ref.extractall("div2k")
