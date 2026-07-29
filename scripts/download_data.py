import click
import copernicusmarine as cm
import dotenv
from hydra import compose, initialize
from pathlib import Path
import geopandas as gpd
import pandas as pd
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
import os
import json

dotenv.load_dotenv()
USERNAME = dotenv.get_key('.env', 'USERNAME')  # USERNAME
PASSWORD = dotenv.get_key('.env', 'PASSWORD')  # PASSWORD

@click.command()
@click.option('--config-name', default='main')
def download_data(config_name, louis):

    with initialize(version_base=None, config_path="../conf"):
        config = compose(config_name=config_name, overrides=[])
    if cm.login(username=USERNAME, password=PASSWORD, force_overwrite=True):
        print("Marine Copernicus login successful")

        zip_file = Path(config.paths.plataformas_shp)
        zip_file_bc  = r'zip://'+str(zip_file.absolute()) + r'!Bacias/PLATAFORMAS_20230919_BC.shp'
        platforms_bc = gpd.read_file(zip_file_bc).to_crs(epsg=4326)

        zip_file_bs = r'zip://'+str(zip_file.absolute()) + r'!Bacias/PLATAFORMAS_20230919_BS.shp'
        platforms_bs = gpd.read_file(zip_file_bs).to_crs(epsg=4326)

        platforms = pd.concat([platforms_bc, platforms_bs])

        min_long, min_lat, max_long, max_lat = platforms.geometry.buffer(config.copernicusmarine.buffer_dg).total_bounds

        cm.subset(
            dataset_id=config.copernicusmarine.water_dataset_id,
            minimum_longitude=config.copernicusmarine.min_long,
            maximum_longitude=config.copernicusmarine.max_long,
            minimum_latitude=config.copernicusmarine.min_lat,
            maximum_latitude=config.copernicusmarine.max_lat,
            start_datetime=str(config.copernicusmarine.start_datetime),
            end_datetime=str(config.copernicusmarine.end_datetime),
            output_directory=Path(config.copernicusmarine.water_dataset_path).parent,
            output_filename=Path(config.copernicusmarine.water_dataset_path).name,
            variables=['uo', 'vo'],
        )

### LOUIS: SAME FUNCTION WITH TINY ADJUSTMENTS ###

user_dict = {}
user = os.getenv("USERNAME") or os.getenv("username") or Path.home().name
repo_loginfile = Path(__file__).resolve().parents[1] / f"copernicus_login_{user}.json"
home_loginfile = Path.home() / f"copernicus_login_{user}.json"
env_loginfile = os.getenv("COPERNICUS_LOGIN_FILE")
login_candidates = [Path(env_loginfile)] if env_loginfile else []
login_candidates.extend([repo_loginfile, home_loginfile])
loginfile = next((path for path in login_candidates if path.exists()), None)
if not loginfile:
    print("Erro! Arquivo de login do Copernicus nao encontrado.")
    print("Procurei em:")
    for path in login_candidates:
        print(f" - {path}")
    quit()
with open(loginfile, "r", encoding="utf-8") as file:
    user_dict = json.load(file)
if not user_dict.get("user") or not user_dict.get("pwd"):
    print("Erro! Arquivo com login do Copernicus nao lido corretamente.")
    quit()
USERNAME = user_dict["user"]
PASSWORD = user_dict["pwd"]


@click.command()
@click.option('--config-name', default='main')
@click.option('--environment', type=str, default="2019")
def download_data_validation(config_name, environment):
    with initialize(version_base=None, config_path="../conf"):
        config = compose(
            config_name=config_name,
            overrides=[f"environment=\"{environment}\""],
        )
    
    if cm.login(username=USERNAME, password=PASSWORD, force_overwrite=True):
        print("Marine Copernicus login successful")
        if os.path.exists(config.copernicusmarine.specificities.water_dataset_path):
            print("1/2 Um arquivo de correnteza já existe com esse nome. Por favor apague-o para poder recriá-lo")
        else:
            cm.subset(
                dataset_id=config.copernicusmarine.specificities.water_dataset_id, #Choose the corresponding environment id here
                minimum_longitude=config.copernicusmarine.min_long,
                maximum_longitude=config.copernicusmarine.max_long,
                minimum_latitude=config.copernicusmarine.min_lat,
                maximum_latitude=config.copernicusmarine.max_lat,
                start_datetime=str(config.copernicusmarine.specificities.start_datetime),
                end_datetime=str(config.copernicusmarine.specificities.end_datetime),
                output_directory=Path(config.copernicusmarine.specificities.water_dataset_path).parent,
                output_filename=Path(config.copernicusmarine.specificities.water_dataset_path).name,
                variables=['uo', 'vo'],
            )
            print("1/2 O arquivo de correnteza foi importado com sucesso")

        if os.path.exists(config.copernicusmarine.specificities.sal_temp_dataset_path):
            print("2/2 Um arquivo de temperatura e salinidade já existe com esse nome. Por favor apague-o para poder recriá-lo")
        else:
            cm.subset(
            dataset_id=config.copernicusmarine.specificities.sal_temp_dataset_id, #Choose the corresponding environment id here
            minimum_longitude=config.copernicusmarine.min_long,
            maximum_longitude=config.copernicusmarine.max_long,
            minimum_latitude=config.copernicusmarine.min_lat,
            maximum_latitude=config.copernicusmarine.max_lat,
            start_datetime=str(config.copernicusmarine.specificities.start_datetime),
            end_datetime=str(config.copernicusmarine.specificities.end_datetime),
            output_directory=Path(config.copernicusmarine.specificities.sal_temp_dataset_path).parent,
            output_filename=Path(config.copernicusmarine.specificities.sal_temp_dataset_path).name,
            variables=['so', 'thetao'],
        )
            print("2/2 O arquivo de temperatura e salinidad foi importado com sucesso")
        

if __name__ == '__main__':
    download_data_validation()
