"""

"""

import copy
import re
from collections import defaultdict
from functools import lru_cache
import math
import pandas as pd
import prettytable

import yaml

from .export import biosphere_flows_dictionary
from .filesystem_constants import VARIABLES_DIR
from .logger import create_logger
from .transformation import (
    BaseTransformation,
    Dict,
    IAMDataCollection,
    InventorySet,
    List,
    Tuple,
    find_fuel_efficiency,
    get_suppliers_of_a_region,
    np,
    uuid,
    ws,
)
from .utils import (
    get_efficiency_solar_photovoltaics,
    get_water_consumption_factors,
    rescale_exchanges,
    get_delimiter
)
from .filesystem_constants import DATA_DIR
from .validation import ElectricityValidation

CAPACITY_FACTORS_WIND = DATA_DIR / "renewables" / "wind_capacity_factors.csv"

logger = create_logger("electricity")



def _update_renewables(
    scenario,
    version,
    system_model,
    use_absolute_efficiency,
):
    windturbines = WindTurbines(
        database=scenario["database"],
        iam_data=scenario["iam data"],
        model=scenario["model"],
        pathway=scenario["pathway"],
        year=scenario["year"],
        version=version,
        system_model=system_model,
        use_absolute_efficiency=use_absolute_efficiency,
        cache=scenario.get("cache"),
        index=scenario.get("index"),
    )

    windturbines.get_component_masses(turbine_type="onshore")
    windturbines.get_component_masses(turbine_type="offshore")

    #windturbines.relink_datasets()
    #scenario["database"] = windturbines.database
    #scenario["index"] = windturbines.index
    #scenario["cache"] = windturbines.cache

    
    return scenario

def get_capacity_factors():

    dataframe = pd.read_csv(
        CAPACITY_FACTORS_WIND, sep=get_delimiter(filepath=CAPACITY_FACTORS_WIND)
    )

    # Convert the DataFrame to an xarray Dataset
    array = dataframe.set_index(["country", "type", ])[
        "capacity factor"
    ].to_xarray()

    return array


def get_power_from_year(year: int, type: str) -> float:
    """
    Return fleet average power (in kW) of wind tubrine based on type (offshore/onshore) and year.
    """

    if type=="onshore":
        return np.clip(0.1438 * year - 287.06, None, 8000)
    
    else:
        return np.clip(0.4665 * year - 934.33, None, 20000)
    

def get_foundation_mass_from_power(power: int, type: str) -> float:
    """
    Return foundation mass (in tons) based on power and foundation type.
    """
    if type=="onshore":
        return np.clip(0.0794 * power + 604.51, None, 2500)
    
    else:
        return np.clip(0.0794 * power + 604.51, None, 2500)
    

def get_tower_mass_from_power(power: int, type: str) -> float:
    """
    Return tower mass (in tons) based on power and foundation type.
    """
    if type=="onshore":
        return np.clip(0.0903 * power + 3.6486, None, 1200)
    
    else:
        return np.clip(0.0653 * power + 19.044, None, 1500)
    
def get_nacelle_mass_from_power(power: int, type: str) -> float:
    """
    Return nacelle mass (in tons) based on power and foundation type.
    """
    if type=="onshore":
        return np.clip(0.0376 * power - 0.8092, None, 400)
    
    else:
        return np.clip(0.0486 * power - 25.633 , None, 1000)

    
def get_rotor_mass_from_power(power: int, type: str) -> float:
    """
    Return rotor mass (in tons) based on power and foundation type.
    """
    if type=="onshore":
        return np.clip(0.0246 * power - 2.8149, None, 250)
    
    else:
        return np.clip(0.0267 * power - 10.29, None, 500)
    
def get_electricity_production(capacity_factor: float, power: int, lifetime: int) -> float:
    """
    Return lifetime electricity production
    """

    return power * capacity_factor * 24 * 365 * lifetime


def get_components_mass_shares(installation_type: str) -> Dict[str, float]:

    if installation_type == "offshore":
        filepath = DATA_DIR / "renewables" / "components_mass_shares_offshore.csv"
    else:
        filepath = DATA_DIR / "renewables" / "components_mass_shares_onshore.csv"

    dataframe = pd.read_csv(
        filepath, sep=get_delimiter(filepath=filepath)
    )

    return dataframe.fillna(0)


class WindTurbines(BaseTransformation):
    """
    Class that modifies electricity markets in the database based on IAM output data.
    Inherits from `transformation.BaseTransformation`.

    :ivar database: wurst database, which is a list of dictionaries
    :vartype database: list
    :ivar iam_data: IAM data
    :vartype iam_data: xarray.DataArray
    :ivar model: name of the IAM model (e.g., "remind", "image")
    :vartype model: str
    :vartype pathway: str
    :ivar year: year of the pathway (e.g., 2030)
    :vartype year: int

    """

    def __init__(
        self,
        database: List[dict],
        iam_data: IAMDataCollection,
        model: str,
        pathway: str,
        year: int,
        version: str,
        system_model: str,
        use_absolute_efficiency: bool = False,
        cache: dict = None,
        index: dict = None,
    ) -> None:
        super().__init__(
            database,
            iam_data,
            model,
            pathway,
            year,
            version,
            system_model,
            cache,
            index,
        )

        self.capacity_factors = get_capacity_factors()

    def get_component_masses(self, turbine_type):

        power = get_power_from_year(self.year, turbine_type)
        foundation = get_foundation_mass_from_power(power, turbine_type)
        tower = get_tower_mass_from_power(power, turbine_type)
        nacelle = get_nacelle_mass_from_power(power, turbine_type)
        rotor = get_rotor_mass_from_power(power, turbine_type)

        # scale up original ecoinvent datasets

        # we start with offshore wind turbines
        # we first look at the dataset representing the fixed parts

        offshore_fixed = copy.deepcopy(ws.get_one(
            self.database,
            ws.equals("name", "wind power plant construction, 2MW, offshore, fixed parts"),
            ws.equals("unit", "unit"),
        ))

        offshore_moving = copy.deepcopy(ws.get_one(
            self.database,
            ws.equals("name", "wind power plant construction, 2MW, offshore, moving parts"),
            ws.equals("unit", "unit"),
        ))

        offshore_fixed["name"] = f"wind power plant construction, {'{:.1f}'.format(power/1000)}MW, offshore, fixed parts"
        offshore_moving["name"] = f"wind power plant construction, {'{:.1f}'.format(power/1000)}MW, offshore, moving parts"
        offshore_fixed["reference product"] = f"wind power plant construction, {'{:.1f}'.format(power/1000)}MW, offshore, fixed parts"
        offshore_moving["reference product"] = f"wind power plant construction, {'{:.1f}'.format(power/1000)}MW, offshore, moving parts"

        for exc in ws.production(
            offshore_fixed,
        ):
            exc["name"] = offshore_fixed["name"]
            exc["product"] = offshore_fixed["reference product"]

        for exc in ws.production(
            offshore_moving,
        ):
            exc["name"] = offshore_moving["name"]
            exc["product"] = offshore_moving["reference product"]


        offshore_shares = get_components_mass_shares("offshore")

        COLUMNS_FIXED = [
            #"nacelle",
            #"rotor",
            #"other",
            #"transformer + cabinet",
            "foundation",
            "tower",
            "platform",
            "grid connector",
        ]

        foundation_mass, tower_mass, platform_mass, grid_connector_mass = 0, 0, 0, 0

        for exc in ws.technosphere(offshore_fixed):
            shares = offshore_shares.loc[
                (offshore_shares["activity"] == exc["name"])
                &(offshore_shares["reference product"] == exc["product"])
                &(offshore_shares["location"] == exc["location"])
                &(offshore_shares["part"] == "fixed")
            ]

            original_amount = copy.deepcopy(exc["amount"])

            if shares[COLUMNS_FIXED].sum().sum() > 0:
                if shares["foundation"].values[0] > 0:
                    foundation_mass += (original_amount * shares["foundation"].values[0])
                if shares["tower"].values[0] > 0:
                    tower_mass += (original_amount * shares["tower"].values[0])
                if shares["platform"].values[0] > 0:
                    platform_mass += (original_amount * shares["platform"].values[0])
                if shares["grid connector"].values[0] > 0:
                    grid_connector_mass += (original_amount * shares["grid connector"].values[0])

        COLUMNS_MOVING = [
            "nacelle",
            "rotor",
            "other",
            "transformer + cabinet",
            # "foundation",
            # "tower",
            # "platform",
            # "grid connector",
        ]

        nacelle_mass, rotor_mass, other_mass, transformer + cabinet_mass = 0, 0, 0, 0

        for exc in ws.technosphere(offshore_moving):
            shares = offshore_shares.loc[
                (offshore_shares["activity"] == exc["name"])
                & (offshore_shares["reference product"] == exc["product"])
                & (offshore_shares["location"] == exc["location"])
                & (offshore_shares["part"] == "moving")
                ]

            original_amount = copy.deepcopy(exc["amount"])

            if shares[COLUMNS_MOVING].sum().sum() > 0:
                if shares["nacelle"].values[0] > 0:
                    nacelle_mass += (original_amount * shares["nacelle"].values[0])
                if shares["rotor"].values[0] > 0:
                    rotor_mass += (original_amount * shares["rotor"].values[0])
                if shares["other"].values[0] > 0:
                    other_mass += (original_amount * shares["other"].values[0])
                if shares["transformer + cabinet"].values[0] > 0:
                    transformer + cabinet_mass += (original_amount * shares["transformer + cabinet"].values[0])

        print(f"Foundation mass: {foundation_mass}")
        print(f"Tower mass: {tower_mass}")
        print(f"Platform mass: {platform_mass}")
        print(f"Grid connector mass: {grid_connector_mass}")
        print(f"Nacelle mass: {nacelle_mass}")
        print(f"Rotor mass: {rotor_mass}")
        print(f"Other mass: {other_mass}")
        print(f"Transformer + cabinet: {transformer + cabinet}")





        results = []

        for country in self.capacity_factors.coords["country"].values:
            if np.isnan(self.capacity_factors.sel(country=country, type=turbine_type).values):
                cf = self.capacity_factors.sel(country=country, type="all").values
            else:
                cf = self.capacity_factors.sel(country=country, type=turbine_type).values

            production = int(get_electricity_production(
                capacity_factor=cf/100,
                power=power,
                lifetime=20
            ))

            print(country)

            try:

                electricity_ds = copy.deepcopy(ws.get_one(
                    self.database,
                    ws.equals("name", "electricity production, wind, 1-3MW turbine, offshore"),
                    ws.equals("location", country),
                ))

                # modify the name of the dataset
                electricity_ds["name"] = f"electricity production, wind, {'{:.1f}'.format(power/1000)}MW turbine, {turbine_type}"
                electricity_ds["reference product"] = f"electricity production, wind, {'{:.1f}'.format(power/1000)}MW turbine, {turbine_type}"

                # modify the production xchange name
                for exc in ws.production(electricity_ds):
                    exc["name"] = electricity_ds["name"]
                    exc["product"] = electricity_ds["reference product"]

                # we replace the inputs of wind turbines (fixed and moving parts) with the new datasets
                for exc in ws.technosphere(
                        electricity_ds,
                    ws.equals("unit", "unit")

                ):
                    exc["name"] = exc["name"].replace("1-3MW", f"{'{:.1f}'.format(power/1000)}MW")
                    exc["product"] = exc["product"].replace("1-3MW", f"{'{:.1f}'.format(power/1000)}MW")
                    exc["amount"] = 1/production

                self.database.append(electricity_ds)

            except:
                pass


            results.append([country, cf, turbine_type, production, production/20])

        table = prettytable.PrettyTable()
        table.field_names = ["Country", "Capacity factor", "Type", "Lifetime prod [kWh]", "Annual prod [kWh]"]
        for result in results:
            table.add_row(result)
        print(table)


    def write_log(self, dataset, status="created"):
        """
        Write log file.
        """

        logger.info(
            f"{status}|{self.model}|{self.scenario}|{self.year}|"
            f"{dataset['name']}|{dataset['location']}|"
            f"{dataset.get('log parameters', {}).get('old efficiency', '')}|"
            f"{dataset.get('log parameters', {}).get('new efficiency', '')}|"
            f"{dataset.get('log parameters', {}).get('transformation loss', '')}|"
            f"{dataset.get('log parameters', {}).get('distribution loss', '')}|"
            f"{dataset.get('log parameters', {}).get('renewable share', '')}|"
            f"{dataset.get('log parameters', {}).get('ecoinvent original efficiency', '')}|"
            f"{dataset.get('log parameters', {}).get('Oberschelp et al. efficiency', '')}|"
            f"{dataset.get('log parameters', {}).get('efficiency change', '')}|"
            f"{dataset.get('log parameters', {}).get('CO2 scaling factor', '')}|"
            f"{dataset.get('log parameters', {}).get('SO2 scaling factor', '')}|"
            f"{dataset.get('log parameters', {}).get('CH4 scaling factor', '')}|"
            f"{dataset.get('log parameters', {}).get('NOx scaling factor', '')}|"
            f"{dataset.get('log parameters', {}).get('PM <2.5 scaling factor', '')}|"
            f"{dataset.get('log parameters', {}).get('PM 10 - 2.5 scaling factor', '')}|"
            f"{dataset.get('log parameters', {}).get('PM > 10 scaling factor', '')}"
        )
