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
        return np.clip(74.923 * year - 148319, None, 8000)
    
    else:
        return np.clip(366.95 * year - 733089, None, 20000)
    

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
        return np.clip(5e-6 * math.pow(power, 2) + 0.086 * power + 1.254, None, 1200)
    
    else:
        return np.clip(0.0618 * math.pow(power, 0.9944), None, 1500)
    
def get_nacelle_mass_from_power(power: int, type: str) -> float:
    """
    Return nacelle mass (in tons) based on power and foundation type.
    """
    if type=="onshore":
        return np.clip(0.0362 * power + 1.1673, None, 400)
    
    else:
        return np.clip(-7e-7 * math.pow(power, 2) + 0.0554 * power - 38.061 , None, 1000)

    
def get_rotor_mass_from_power(power: int, type: str) -> float:
    """
    Return rotor mass (in tons) based on power and foundation type.
    """
    if type=="onshore":
        return np.clip(0.0247 * power - 2.6492, None, 250)
    
    else:
        return np.clip(0.0281 * power - 14.862, None, 500)
    
def get_electricity_production(capacity_factor: float, power: int, lifetime: int) -> float:
    """
    Return lifetime electricity production
    """

    return power * capacity_factor * 24 * 365 * lifetime


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

        

        print(f"{self.year} - {power} - foundation: {foundation} - tower: {tower} - nacelle: {nacelle} - rotor: {rotor}")

        results = []

        print(self.capacity_factors.sel(country="AT"))


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
