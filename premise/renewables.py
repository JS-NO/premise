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
import csv
from scipy.stats import  truncnorm
from scipy.optimize import minimize

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
    scenario["database"] = windturbines.database
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
        # return np.clip(0.0581 * (power ** 1.0626), None, 1200) #linear
        return np.clip(0.0903 * power + 3.6486, None, 1200)
    
    else:
        #return np.clip((-4e-6 * power**2) + (0.1201 * power) - 92.119, None, 1500) #polynomial
        # return np.clip(0.0653 * power + 19.044, None, 1500) #linear
        # return np.clip(0.2097 * (power ** 0.8213), None, 1500) #linear
        return np.clip(0.0653 * power  + 19.044, None, 1500) #linear
    
def get_nacelle_mass_from_power(power: int, type: str) -> float:
    """
    Return nacelle mass (in tons) based on power and foundation type.
    """
    if type=="onshore":
        #return np.clip((2e-6 * power**2) + (0.0291 * power) + 5.8799, None, 400) #polynomial
        # return np.clip(0.0376 * power - 0.8092, None, 400) #linear
        return np.clip(0.0606 * (power ** 0.9278), None, 400) #linear
    
    else:
        return np.clip((-7e-7 * power**2) + (0.0554 * power) - 38.061 , None, 1100) #polynomial
        # return np.clip(0.0486 * power - 25.633, None, 1100) #linear

    
def get_rotor_mass_from_power(power: int, type: str) -> float:
    """
    Return rotor mass (in tons) based on power and foundation type.
    """
    if type=="onshore":
        #return np.clip((-3e-8 * power**2) + (0.0248 * power) - 2.9359, None, 250)
        return np.clip(0.0244 * power - 2.3363, None, 250)
    
    else:
        #return np.clip((-4e-7 * power**2) + (0.03 * power) - 16.055, None, 600)
        return np.clip(0.0281 * power - 14.86, None, 600)
    
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

def get_fleet_distribution(power: int) -> dict:
    """
    Generate a distribution of wind turbine capacities such that
    the weighted average equals the fleet's average capacity.

    Parameters:
    - power (int): Fleet average capacity in kW.

    Returns:
    - dict: Distribution of turbine capacities (bin centers) with percentages.
    """

    # Parameters for the log-normal distribution
    sigma = 0.7  # Standard deviation (controls skewness)
    min_capacity = 1000  # Minimum capacity (kW)
    max_capacity = 25000  # Maximum capacity (kW)

    # Define bin edges and centers (round numbers)
    bin_edges = np.arange(min_capacity, max_capacity + 1000, 1000)
    bin_centers = bin_edges[:-1]

    # Generate truncated log-normal data
    mu = np.log(power) - (sigma ** 2 / 2)
    a, b = (np.log(min_capacity) - mu) / sigma, (np.log(max_capacity) - mu) / sigma
    n_samples = 100000
    lognormal_samples = truncnorm.rvs(a, b, loc=mu, scale=sigma, size=n_samples)
    capacities = np.exp(lognormal_samples)

    # Compute histogram counts
    counts, _ = np.histogram(capacities, bins=bin_edges)

    # Initial raw distribution
    raw_distribution = counts / counts.sum()

    # Optimization target: match the weighted average to the target
    def objective(weights):
        return np.sum((weights / weights.sum()) * bin_centers) - power

    # Constraint: weights must sum to 1
    constraints = {"type": "eq", "fun": lambda w: w.sum() - 1}

    # Bounds: weights must be non-negative
    bounds = [(0, 1) for _ in range(len(bin_centers))]

    # Optimize weights
    result = minimize(
        lambda w: abs(objective(w)),  # Minimize the absolute difference
        x0=raw_distribution,  # Initial guess
        bounds=bounds,
        constraints=constraints,
        method="SLSQP"  # Sequential Least Squares Programming
    )

    if not result.success:
        raise ValueError("Optimization failed to converge.")

    # Extract optimized weights
    optimized_weights = result.x / result.x.sum()

    # Validate the weighted average
    validated_average = np.sum(optimized_weights * bin_centers)

    # Return the final distribution as a dictionary
    return {int(center): optimized_weights[i] for i, center in enumerate(bin_centers)}

def get_wind_power_generation():
    """
    Return wind power generation by country and technology.
    Data from https://pxweb.irena.org/pxweb/en/IRENASTAT
    """

    filepath = DATA_DIR / "renewables" / "wind_power_generation_by_country.csv"
    wind_power_generation = {}

    with open(filepath, "r") as f:
        reader = csv.DictReader(f)
        # skip header
        next(reader)

        for row in reader:

            if row["country code"] not in wind_power_generation:
                wind_power_generation[row["country code"]] = {}

            wind_power_generation[row["country code"]][row["Technology"]] = float(row["generation (GWh)"]) * 1e6

    return wind_power_generation

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

        # power = get_power_from_year(self.year, turbine_type)

        results = []
        created_datasets = []

        for power in range(1000, 26000, 1000):
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

                        if exc.get("uncertainty type") == 2:
                            # log normal distribution
                            exc["loc"] = math.log(exc["amount"])

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

                    if exc.get("uncertainty type") == 2:
                        # log normal distribution
                        exc["loc"] = math.log(exc["amount"])

            # add the new dataset to the database
            self.database.append(fixed)
            # add the new dataset to the database
            self.database.append(moving)



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

                    created_datasets.append(
                        (country, power)
                    )

                except:
                    pass

                results.append([country, cf, turbine_type, production, production/20])

        fleet_average_power = get_power_from_year(self.year, turbine_type)
        # create a capacity distribution base don the fleet average capacity
        fleet_distribution = get_fleet_distribution(
            fleet_average_power
        )

        # create, for each country, a fleet average dataset for the wind turbines
        # considering the fleet capacity distribution

        wind_power_gen = get_wind_power_generation()

        for country in self.capacity_factors.coords["country"].values:
            if len([x for x in created_datasets if x[0] == country]) == 0:
                # no wind turbine dataset created for this country
                # e.g., offshore market for landlocked countries
                continue

            new_market_dataset = {
                "name": f"market for electricity, from wind turbine, {turbine_type}",
                "location": country,
                "reference product": f"electricity, high voltage",
                "unit": "kilowatt hour",
                "code": str(uuid.uuid4().hex),
                "comment": f"Assumed fleet average power: {fleet_average_power} kW.",
                "exchanges": [
                    {
                        "amount": fleet_distribution[p],
                        "type": "technosphere",
                        "unit": "kilowatt hour",
                        "name": f"electricity production, wind, {'{:.1f}'.format(p/1000)}MW turbine, {turbine_type}",
                        "product": f"electricity, high voltage",
                        "location": country,
                        "uncertainty type": 0,
                    }
                    for p in fleet_distribution
                    if (country, p) in created_datasets
                ],
            }

            # normalize the exchanges
            total = sum([exc["amount"] for exc in new_market_dataset["exchanges"]])
            for exc in new_market_dataset["exchanges"]:
                exc["amount"] /= total

            # add production exchange
            new_market_dataset["exchanges"].extend([
                {
                    "amount": 1,
                    "type": "production",
                    "unit": "kilowatt hour",
                    "name": new_market_dataset["name"],
                    "product": new_market_dataset["reference product"],
                    "location": new_market_dataset["location"],
                    "uncertainty type": 0,
                    "production volume": wind_power_gen.get(country, {}).get(turbine_type, 0),
                }
            ])

            self.database.append(new_market_dataset)


    def write_log(self, dataset, status="created"):
        """
        Write log file.
        """

        logger.info(
            f"{status}|{self.model}|{self.scenario}|{self.year}|"
            f"{dataset['name']}|{dataset['location']}|"
        )
