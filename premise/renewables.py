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

    windturbines.create_wind_turbine_datasets(turbine_type="offshore")
    windturbines.create_wind_turbine_datasets(turbine_type="onshore")

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


def get_power_from_year(year: int, type: str) -> int:
    """
    Return fleet average power (in kW) of wind tubrine based on
    type (offshore/onshore) and year.
    """

    if type=="onshore":
        return int(np.clip(0.1438 * year - 287.06, None, 8000) * 1000)
    
    else:
        return int(np.clip(0.4665 * year - 934.33, None, 20000) * 1000)
    

def get_foundation_mass_from_power(power: int, type: str) -> float:
    """
    Return foundation mass (in tons) based on power and foundation type.
    """
    if type=="onshore":
        return np.clip(0.3873 * power + 108.8, None, 3500) #linear
        #return np.clip((2e-6 * power**2) + (0.3702 * power) + 137.56, None, 3500) #polynomial
    
    else:
        return np.clip(0.1327 * power - 28.602, None, 7000) #linear
        #return np.clip((2e-6 * power**2) + (0.0944 * power) + 106.23, None, 7000) #polynomial
    

def get_tower_mass_from_power(power: int, type: str) -> float:
    """
    Return tower mass (in tons) based on power and foundation type.
    """
    if type=="onshore":
        #return np.clip((-4e-6 * power**2) + (0.1105 * power) - 8.5339, None, 1200) #polynomial
        return np.clip(0.0903 * power + 3.6486, None, 1200) #linear
    
    else:
        #return np.clip((-4e-6 * power**2) + (0.1201 * power) - 92.119, None, 1500) #polynomial
        return np.clip(0.0653 * power + 19.044, None, 1500) #linear
    
def get_nacelle_mass_from_power(power: int, type: str) -> float:
    """
    Return nacelle mass (in tons) based on power and foundation type.
    """
    if type=="onshore":
        #return np.clip((2e-6 * power**2) + (0.0291 * power) + 5.8799, None, 400) #polynomial
        return np.clip(0.0376 * power - 0.8092, None, 400) #linear
    
    else:
        #return np.clip((-7e-7 * power**2) + (0.0556 * power) - 38.344 , None, 1100) #polynomial
        return np.clip(0.0486 * power - 25.633, None, 1100) #linear

    
def get_rotor_mass_from_power(power: int, type: str) -> float:
    """
    Return rotor mass (in tons) based on power and foundation type.
    """
    if type=="onshore":
        #return np.clip((-3e-8 * power**2) + (0.0248 * power) - 2.9359, None, 250)
        return np.clip(0.0246 * power - 2.8149, None, 250)
    
    else:
        #return np.clip((-4e-7 * power**2) + (0.03 * power) - 16.055, None, 600)
        return np.clip(0.0267 * power - 10.29, None, 600)
    
def get_electricity_production(capacity_factor: float, power: int, lifetime: int) -> float:
    """
    Return lifetime electricity production
    """

    return power * capacity_factor * 24 * 365 * lifetime


def get_components_mass_shares(installation_type: str) -> pd.DataFrame:

    if installation_type == "offshore":
        filepath = DATA_DIR / "renewables" / "components_mass_shares_offshore.csv"
    else:
        filepath = DATA_DIR / "renewables" / "components_mass_shares_onshore.csv"

    dataframe = pd.read_csv(
        filepath, sep=get_delimiter(filepath=filepath)
    )

    return dataframe.fillna(0)

def create_new_dataset(dataset, power):

    new_dataset = copy.deepcopy(dataset)

    new_dataset["name"] = new_dataset["name"].replace("2MW", f"{'{:.1f}'.format(power/1000)}MW").replace("800kW", f"{'{:.1f}'.format(power/1000)}MW")
    new_dataset["reference product"] = new_dataset["reference product"].replace("2MW", f"{'{:.1f}'.format(power/1000)}MW").replace("800kW", f"{'{:.1f}'.format(power/1000)}MW")
    new_dataset["code"] = str(uuid.uuid4().hex)

    new_dataset["comment"] = f"Adapted from {dataset['name']} for a {power} kW wind turbine."

    for exc in ws.production(new_dataset):
        exc["name"] = new_dataset["name"]
        exc["product"] = new_dataset["reference product"]
        if "input" in exc:
            del exc["input"]

    return new_dataset


def get_current_masses_from_dataset(dataset, shares, components, components_type) -> Dict[str, float]:

    components_masses = {component: 0 for component in components}

    for exc in ws.technosphere(dataset):
        df_share = shares.loc[
            (shares["activity"] == exc["name"])
            &(shares["reference product"] == exc["product"])
            &(shares["location"] == exc["location"])
            &(shares["part"] == components_type)
        ]

        if df_share[components].sum().sum() > 0:
            for component in components:
                if df_share[component].values[0] > 0:
                    if exc["amount"]>0:
                        if "concrete" in exc["name"] and exc["unit"] == "cubic meter":
                            factor = 2400 #kg/m3
                        else:
                            factor = 1
                        components_masses[component] += (exc["amount"] * df_share[component].values[0] * factor)

    return components_masses


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

    def get_target_component_masses(self, turbine_type) -> Dict[str, float]:

        power = get_power_from_year(self.year, turbine_type) # in kW
        foundation = get_foundation_mass_from_power(power, turbine_type) * 1000 # in kg
        tower = get_tower_mass_from_power(power, turbine_type) * 1000 # in kg
        nacelle = get_nacelle_mass_from_power(power, turbine_type) * 1000 # in kg
        rotor = get_rotor_mass_from_power(power, turbine_type) * 1000 # in kg
        #return grid as a fixed target mass
        if turbine_type == "offshore":
            grid_connector = 92000  # in kg
        else:
            grid_connector = 19000  # in kg

        return {"foundation": foundation, "tower": tower, "nacelle": nacelle, "rotor": rotor, "grid connector": grid_connector}

    def create_wind_turbine_datasets(self, turbine_type):
        """
        Create new datasets for wind turbines based on the power and turbine type.
        """

        # scale up original ecoinvent datasets

        # we start with offshore wind turbines
        # we first look at the dataset representing the fixed parts

        power = get_power_from_year(self.year, turbine_type)

        if turbine_type == "onshore":
            dataset_name_to_copy = "wind power plant construction, 800kW, fixed parts"
        else:
            dataset_name_to_copy = "wind power plant construction, 2MW, offshore, fixed parts"

        fixed = create_new_dataset(
            ws.get_one(
                self.database,
                ws.equals("name", dataset_name_to_copy),
                ws.equals("unit", "unit"),
            ),
            power
        )

        if turbine_type == "onshore":
            dataset_name_to_copy = "wind power plant construction, 800kW, moving parts"
        else:
            dataset_name_to_copy = "wind power plant construction, 2MW, offshore, moving parts"

        moving = create_new_dataset(
            ws.get_one(
                self.database,
                ws.equals("name", dataset_name_to_copy),
                ws.equals("unit", "unit"),
            ),
            power
        )

        components_shares = get_components_mass_shares(turbine_type)

        COLUMNS = {
            "fixed": [
                "foundation",
                "tower",
                "platform",
                #"grid connector"
            ],
            "moving": [
                "nacelle",
                "rotor",
                "other",
                "transformer + cabinet",
            ]
        }

        #added because the grid connection is a part of the moving parts for the 800kW onshore wind turbine.
        if turbine_type == "onshore":
            COLUMNS["moving"].append("grid connector")
        elif turbine_type == "offshore":
            COLUMNS["fixed"].append("grid connector")

        current_component_masses = get_current_masses_from_dataset(fixed, components_shares, COLUMNS["fixed"], "fixed")
        current_component_masses.update(get_current_masses_from_dataset(moving, components_shares, COLUMNS["moving"], "moving"))
        target_component_masses = self.get_target_component_masses(turbine_type)

        scaling_factors = {
            component: target_component_masses.get(component, 0) / current_component_masses.get(component, 1)
            for component in COLUMNS["fixed"] + COLUMNS["moving"]
                             if current_component_masses.get(component, 1) > 0
        }

        print(turbine_type)
        print(scaling_factors)
        print(f" current component masses: {current_component_masses}")
        print(f" target component masses: {target_component_masses}")

        for components_type, dataset in {
            "moving": moving,
            "fixed": fixed,
        }.items():
            for exc in ws.technosphere(
                dataset,
            ):
                df_share = components_shares.loc[
                    (components_shares["activity"] == exc["name"])
                    &(components_shares["reference product"] == exc["product"])
                    &(components_shares["location"] == exc["location"])
                    &(components_shares["part"] == components_type)
                ]

                weighted_scaling_factor = 0

                if df_share[COLUMNS[components_type]].sum().sum() > 0:
                    for component in COLUMNS[components_type]:
                        if df_share[component].values[0] > 0:
                            weighted_scaling_factor += scaling_factors[component] * df_share[component].values[0]

                if weighted_scaling_factor > 0:
                    exc["amount"] *= weighted_scaling_factor
                    exc["comment"] = f"Original amount: {exc['amount'] / weighted_scaling_factor}. Scaling factor: {weighted_scaling_factor}."

            dataset["comment"] += f" Scaling factors: {scaling_factors}."


        # add 0.5 kWh electricity consumption per kg of material
        # in the moving parts
        # mass of moving parts
        mass_moving = sum(target_component_masses.get(component, 0) for component in COLUMNS["moving"])
        # electricity consumption
        electricity_consumption = 0.5 * mass_moving

        for exc in ws.technosphere(moving):
            if "electricity" in exc["name"] and exc["unit"] == "kilowatt hour":
                exc["comment"] = f"Original amount: {exc['amount']}. New amount: {electricity_consumption}. 0.5 kWh per kg of material in the moving parts. Mass of moving parts: {mass_moving} kg."
                exc["amount"] = electricity_consumption

        # add the new dataset to the database
        self.database.append(fixed)
        # add the new dataset to the database
        self.database.append(moving)


        results = []

        for country in self.capacity_factors.coords["country"].values:
            if np.isnan(self.capacity_factors.sel(country=country, type=turbine_type).values):
                cf = self.capacity_factors.sel(country=country, type="all").values
            else:
                cf = self.capacity_factors.sel(country=country, type=turbine_type).values

            production = int(get_electricity_production(
                capacity_factor=cf / 100,
                power=int(power),
                lifetime=20
            ))

            try:
                if turbine_type == "onshore":
                    dataset_name = "electricity production, wind, <1MW turbine, onshore"
                else:
                    dataset_name = "electricity production, wind, 1-3MW turbine, offshore"

                electricity_ds = copy.deepcopy(ws.get_one(
                    self.database,
                    ws.equals("name", dataset_name),
                    ws.equals("location", country),
                ))

                # modify the name of the dataset
                electricity_ds["name"] = f"electricity production, wind, {'{:.1f}'.format(power/1000)}MW turbine, {turbine_type}"
                electricity_ds["reference product"] = f"electricity, high voltage"
                electricity_ds["code"] = str(uuid.uuid4().hex)
                electricity_ds["comment"] = (f"Generated from {dataset_name} for a {power} kW wind turbine."
                                             f"Assumed lifetime: 20 years. Capacity factor: {cf}%.")

                # modify the production exchange name
                for exc in ws.production(electricity_ds):
                    exc["name"] = electricity_ds["name"]
                    exc["product"] = electricity_ds["reference product"]
                    if "input" in exc:
                        del exc["input"]

                # let's remove the current wind turbine inputs
                electricity_ds["exchanges"] = [
                    exc for exc in electricity_ds["exchanges"]
                    if exc["unit"] != "unit"
                ]

                # we need to scale up the use of oil by the ratio between the new and old rated power
                for exc in ws.technosphere(electricity_ds):
                    if "oil" in exc["name"]:
                        exc["amount"] *= power / (2000 if turbine_type == "offshore" else 800)

                electricity_ds["exchanges"].extend([
                    {
                        "amount": 1/production,
                        "type": "technosphere",
                        "unit": "unit",
                        "name": fixed["name"],
                        "product": fixed["reference product"],
                        "location": fixed["location"],
                        "uncertainty type": 0,
                        "comment": f"Lifetime production {production} kWh",
                    },
                    {
                        "amount": 1/production,
                        "type": "technosphere",
                        "unit": "unit",
                        "name": moving["name"],
                        "product": moving["reference product"],
                        "location": moving["location"],
                        "uncertainty type": 0,
                        "comment": f"Lifetime production {production} kWh",
                    }
                ])

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
