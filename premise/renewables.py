"""

"""

import copy
import re
from collections import defaultdict
from functools import lru_cache
import math
import pandas as pd
import prettytable
import wurst

import yaml
import csv
from scipy.stats import  truncnorm
from scipy.optimize import minimize
import scipy.stats as stats
from scipy.optimize import fsolve
import scipy.integrate as integrate

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
STEEL_DENSITY = 8000  # kg/m3
COPPER_DENSITY = 8960  # kg/m3

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

    scenario["database"] = windturbines.database
    scenario["index"] = windturbines.index
    scenario["cache"] = windturbines.cache

    
    return scenario

def get_capacity_factors() -> pd.DataFrame:

    return pd.read_csv(
        CAPACITY_FACTORS_WIND,
        sep=get_delimiter(filepath=CAPACITY_FACTORS_WIND),
        keep_default_na=False, na_values=['']
    )


def get_power_from_year(year: int, type: str) -> int:
    """
    Return fleet average power (in kW) of wind tubrine based on
    type (offshore/onshore) and year.
    """

    if type=="onshore":
        return int(np.clip(0.1438 * year - 287.06, None, 8000) * 1000)
    
    else:
        return int(np.clip(0.4665 * year - 934.33, None, 20000) * 1000)


def set_onshore_cable_requirements(
    power,
    tower_height,
    distance_m=550,
    voltage_kv=33,
    power_factor=0.95,
    resistivity_copper=1.68e-8,
    max_voltage_drop_percent=3,
):
    """
    Calculate the required cross-sectional area of a copper cable for a wind turbine connection.

    :param power: Power output of the wind turbine in MW
    :param distance_m: Distance from the wind turbine to the transformer in meters
    :param voltage_kv: Voltage of the cable in kV
    :param power_factor: Power factor of the wind turbine
    :param resistivity_copper: Resistivity of copper in ohm-meters
    :param max_voltage_drop_percent: Maximum allowable voltage drop as a percentage of the voltage
    :return: Copper mass in kg

    """
    # Convert input parameters to standard units
    voltage_v = voltage_kv * 1e3  # Convert kV to V
    max_voltage_drop = (
        max_voltage_drop_percent / 100
    ) * voltage_v  # Maximum voltage drop in volts

    # Calculate current (I) using the formula: I = P / (sqrt(3) * V * PF)
    current_a = (power * 1000) / (3**0.5 * voltage_v * power_factor)

    # Calculate the total cable length (round trip)
    total_length_m = 2 * distance_m

    # Calculate the required resistance per meter to stay within the voltage drop limit
    max_resistance_per_meter = max_voltage_drop / (current_a * total_length_m)

    # Calculate the required cross-sectional area using R = rho / A
    cross_section_area_m2 = resistivity_copper / max_resistance_per_meter

    # Convert cross-sectional area to mm²
    cross_section_area_mm2 = cross_section_area_m2 * 1e6

    copper_mass = cross_section_area_mm2 * total_length_m * 1e-6 * COPPER_DENSITY

    # Also, add the cable inside the wind turbine, which has a 640 mm2 cross-section
    copper_mass += 640 * 1e-6 * tower_height * COPPER_DENSITY

    return copper_mass

def set_offshore_cable_requirements(
    power: int,
    cross_section: float,
    dist_transfo: float,
    dist_coast: float,
    park_size: int,
) -> Tuple[float, float]:
    """
    Return the required cable mass as well as the energy needed to lay down the cable.
    :param power: rotor power output (in kW)
    :param cross_section: cable cross-section (in mm2)
    :param dist_transfo: distance to transformer (in m)
    :param dist_coast: distance to coastline (in m)
    :param park_size:
    :return:
    """

    m_copper = (cross_section * 1e-6 * dist_transfo) * COPPER_DENSITY

    # 450 l diesel/hour for the ship that lays the cable at sea bottom
    # 39 MJ/liter, 15 km/h as speed of laying the cable
    energy_cable_laying_ship = 450 * 39 / 15 * dist_transfo

    # Cross-section calculated based on the farm cumulated power,
    # and the transport capacity of the Nexans cables @ 150kV
    # if the cumulated power of the park cannot be transported @ 33kV

    # Test if the cumulated power of the wind farm is inferior to 30 MW,
    # If so, we use 33 kV cables.

    cross_section_ = np.where(
        power * park_size <= 30e3,
        np.interp(
            power * park_size,
            np.array([352, 399, 446, 502, 581, 652, 726, 811, 904, 993]) * 33,
            np.array([95, 120, 150, 185, 240, 300, 400, 500, 630, 800]),
        ),
        np.interp(
            power * park_size,
            np.array([710, 815, 925, 1045, 1160, 1335, 1425, 1560]) * 150,
            np.array([400, 500, 630, 800, 1000, 1200, 1600, 2000]),
        ),
    )

    m_copper += (
        cross_section_ * 1e-6 * (dist_coast / park_size)
    ) * COPPER_DENSITY

    # 450 l diesel/hour for the ship that lays the cable at sea bottom
    # 39 MJ/liter, 15 km/h as speed of laying the cable
    energy_cable_laying_ship += 450 * 39 / 15 * dist_coast / park_size

    return m_copper, energy_cable_laying_ship * 0.5

def get_pile_mass(power: int, pile_height: float) -> float:
    """
    Return the mass of the steel pile based on the power output of the rotor and the height of the pile.
    :param power: power output (in kW) of the rotor
    :param pile_height: height (in m) of the pile
    :return: mass of the steel pile (in kg)
    """

    # The following lists store data on the relationship
    # between the power output of the rotor and the diameter of the pile.
    # diameters, in meters
    diameter_x = [5, 5.5, 5.75, 6.75, 7.75]
    # kW
    power_y = [3000, 3600, 4000, 8000, 10000]

    # Use polynomial regression to find the function that best fits the data.
    # This function relates the diameter of the pile with the power output of the rotor.
    fit_diameter = np.polyfit(power_y, diameter_x, 1)
    f_fit_diameter = np.poly1d(fit_diameter)

    # Calculate the outer diameter of the pile based on the power output of the rotor.
    outer_diameter = f_fit_diameter(power)

    # Calculate the cross-section area of the pile based on the outer diameter.
    outer_area = (np.pi / 4) * (outer_diameter**2)

    # Calculate the volume of the pile based on the outer area and the pile height.
    outer_volume = outer_area * pile_height

    # Calculate the inner diameter of the pile based on the power output of the rotor and the thickness of the pile.
    inner_diameter = outer_diameter
    pile_thickness = np.interp(
        power,
        [2000, 3000, 3600, 4000, 8000, 10000],
        [0.07, 0.10, 0.13, 0.16, 0.19, 0.22],
    )
    inner_diameter -= 2 * pile_thickness

    # Calculate the cross-section area of the inner part of the pile.
    inner_area = (np.pi / 4) * (inner_diameter**2)

    # Calculate the volume of the inner part of the pile based on the inner area and the pile height.
    inner_volume = inner_area * pile_height

    # Calculate the volume of steel used in the pile by subtracting the inner volume from the outer volume.
    volume_steel = outer_volume - inner_volume

    # Calculate the weight of the steel used in the pile based on its volume and density.
    weight_steel = STEEL_DENSITY * volume_steel

    # Return the weight of the steel pile.
    return weight_steel


def penetration_depth_fit() -> np.poly1d:
    """
    Return a penetration depth fit model of the steel pile of the offshore wind turbine.
    :return:
    """
    # meters
    depth = [22.5, 22.5, 23.5, 26, 29.5]
    # kW
    power = [3000, 3600, 4000, 8000, 10000]
    fit_penetration = np.polyfit(power, depth, 1)
    f_fit_penetration = np.poly1d(fit_penetration)
    return f_fit_penetration


def get_pile_height(power: int, sea_depth: float) -> float:
    """
    Returns undersea pile height (m) from rated power output (kW), penetration depth and sea depeth.
    :param power: power output (kW)
    :param sea_depth: sea depth (m)
    :return: pile height (m)
    """
    fit_penetration_depth = penetration_depth_fit()
    return 9 + fit_penetration_depth(power) + sea_depth


def get_transition_height() -> np.poly1d:
    """
    Returns a fitting model for the height of
    transition piece (in m), based on pile height (in m).
    :return:
    """
    pile_length = [35, 50, 65, 70, 80]
    transition_length = [15, 20, 24, 30, 31]
    fit_transition_length = np.polyfit(pile_length, transition_length, 1)
    return np.poly1d(fit_transition_length)


def get_transition_mass(transition_length: float) -> float:
    """
    Returns the mass of transition piece (in kg).
    :return:
    """
    transition_lengths = [15, 20, 24, 30]
    transition_weight = [150, 200, 260, 370]
    fit_transition_weight = np.polyfit(transition_lengths, transition_weight, 1)

    return np.poly1d(fit_transition_weight)(transition_length) * 1000

def get_foundation_mass_from_power(power: int, type: str) -> float:
    """
    Return foundation mass (in tons) based on power and foundation type.
    """
    if type=="onshore":
        return 0.63 * np.power(power, 0.95)
    
    else:
        return 0.08 * np.power(power, 1.06)

def get_hub_height_from_rotor_diameter(diameter: float) -> float:

    return 3.79 * np.power(diameter, 0.69)

def get_tower_mass_from_power(power: int) -> float:
    """
    Return tower mass (in tons) based on power and hub height.
    """
    return 0.29 * np.power(power, 0.84)

    
def get_nacelle_mass_from_power(power: int) -> float:
    """
    Return nacelle mass (in tons) based on power and foundation type.
    """
    return 0.0249 * np.power(power, 1.0529)


def get_rotor_mass_from_rotor_diameter(diameter):

    return 0.02 * np.power(diameter, 1.73)

def get_rotor_diameter_from_power(power: int) -> float:
    """
    Return rotor diameter (in meters) based on power.
    """
    return 2.18 * np.power(power, 0.49)
    
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

    new_dataset["name"] = new_dataset["name"].replace("2MW", f"{'{:.1f}'.format(int(power/1000))}MW").replace("800kW", f"{'{:.1f}'.format(int(power/1000))}MW")
    new_dataset["reference product"] = new_dataset["reference product"].replace("2MW", f"{'{:.1f}'.format(int(power/1000))}MW").replace("800kW", f"{'{:.1f}'.format(int(power/1000))}MW")
    new_dataset["code"] = str(uuid.uuid4().hex)

    new_dataset["comment"] = ""

    for exc in ws.production(new_dataset):
        exc["name"] = new_dataset["name"]
        exc["product"] = new_dataset["reference product"]
        if "input" in exc:
            del exc["input"]

    new_dataset["exchanges"] = [exc for exc in new_dataset["exchanges"] if exc["type"] == "production"]

    return new_dataset




def get_fleet_distribution(power: int, turbine_type: str) -> dict:
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
    max_capacity = 10000 if turbine_type == "onshore" else 22000  # Maximum capacity (kW)

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
    assert np.isclose(np.sum(optimized_weights * bin_centers), power, rtol=1e-3), f"Weighted average mismatch: {np.sum(optimized_weights * bin_centers)} vs. {power}"

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

    def get_target_component_masses(self, turbine_type: str, power: int) -> Dict[str, float]:

        foundation = get_foundation_mass_from_power(power, turbine_type) * 1000 # in kg
        rotor_diameter = get_rotor_diameter_from_power(power)

        hub_height = get_hub_height_from_rotor_diameter(rotor_diameter)
        tower = get_tower_mass_from_power(power) * 1000 # in kg

        nacelle = get_nacelle_mass_from_power(power) * 1000 # in kg
        rotor = get_rotor_mass_from_rotor_diameter(rotor_diameter) * 1000 # in kg

        sea_depth = 20 #m for offshore
        pile_height = get_pile_height(power, sea_depth)

        if turbine_type == "onshore":
            pile_mass = 0
            transition_mass = 0
        else:
            pile_mass = get_pile_mass(power, pile_height)
            transition_length = get_transition_height()(pile_height)
            transition_mass = get_transition_mass(transition_length)

        # medium-voltage transformer: we assume 1 mega-volt-ampere per 1 MW of power + 10%
        medium_voltage_transformer = 1.1 * power / 1000  # in MVA

        # high-voltage transformer: allocated between number of turbines in the park
        # assumed 10 turbines per park
        turbines_per_park = 10
        high_voltage_transformer = 1 / turbines_per_park

        distance_to_transformer = 200
        distance_to_coast = 2000
        # cable/grid connection
        if turbine_type == "offshore":
            cable_mass = set_offshore_cable_requirements(
                power=power,
                cross_section=1500,
                dist_transfo=distance_to_transformer,
                dist_coast=distance_to_coast,
                park_size=turbines_per_park,
            )[0]

        else:
            cable_mass = set_onshore_cable_requirements(
                power=power,
                tower_height=hub_height,
                distance_m=distance_to_transformer,
            )

        return {
            "foundation": int(foundation),
            "tower": int(tower),
            "nacelle": int(nacelle),
            "rotor": int(rotor),
            "pile": int(pile_mass),
            "transition": int(transition_mass),
            "electronic cabinet": 1,
            "grid connector": int(cable_mass),
            "medium-voltage transformer": medium_voltage_transformer,
            "high-voltage transformer": high_voltage_transformer,
            "rotor diameter": int(rotor_diameter),
            "hub height": int(hub_height),
            "sea depth": int(sea_depth),
            "pile height": int(pile_height),
            "turbines in park": turbines_per_park,
            "distance to transformer": int(distance_to_transformer),
            "distance to coast": int(distance_to_coast),

        }

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
        summary_scaling_factors = []

        max_power = 10000 if turbine_type == "onshore" else 22000

        for power in range(1000, max_power + 1000, 1000):
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

            target_component_masses = self.get_target_component_masses(turbine_type, power)

            components_datasets = {
                "rotor": {
                    "dataset": {
                        "onshore": "rotor production, for onshore wind turbine",
                        "offshore": "rotor production, for offshore wind turbine",
                        "amount": target_component_masses["rotor"],
                    },
                    "eol": {
                        "onshore": "treatment of rotor, for onshore wind turbine",
                        "offshore": "treatment of rotor, for offshore wind turbine",
                        "amount": target_component_masses["rotor"] * -1,
                    },
                },
                "nacelle": {
                    "dataset": {
                        "onshore": "nacelle production, for onshore wind turbine",
                        "offshore": "nacelle production, for offshore wind turbine",
                        "amount": target_component_masses["nacelle"],
                    },
                    "eol": {
                        "onshore": "treatment of nacelle, for onshore wind turbine",
                        "offshore": "treatment of nacelle, for offshore wind turbine",
                        "amount": target_component_masses["nacelle"] * -1,
                    },
                },
                "tower": {
                    "dataset": {
                        "onshore": "tower production, for onshore wind turbine",
                        "offshore": "tower production, for offshore wind turbine",
                        "amount": target_component_masses["tower"],
                    },
                    "eol": {
                        "onshore": "treatment of tower, for onshore wind turbine",
                        "offshore": "treatment of tower, for offshore wind turbine",
                        "amount": target_component_masses["tower"] * -1,
                    },
                },
                "electronic cabinet": {
                    "dataset": {
                        "onshore": "electronic cabinet production, for wind turbine",
                        "offshore": "electronic cabinet production, for wind turbine",
                        "amount": 1,
                    },
                    "eol": {
                        "onshore": "treatment of electronic cabinet, for wind turbine",
                        "offshore": "treatment of electronic cabinet, for wind turbine",
                        "amount": -1,
                    },
                },
                "grid connector": {
                    "dataset": {
                        "onshore": "grid connector production, per kg of copper, for wind turbine",
                        "offshore": "grid connector production, per kg of copper, for wind turbine",
                        "amount": target_component_masses["grid connector"],
                    },
                    "eol": {
                        "onshore": "treatment of grid connector, for wind turbine",
                        "offshore": "treatment of grid connector, for wind turbine",
                        "amount": target_component_masses["grid connector"] * -1,
                    },
                },
                "transition": {
                    "dataset": {
                        "onshore": "platform production, for onshore wind turbine",
                        "offshore": "platform production, for offshore wind turbine",
                        "amount": target_component_masses["transition"]
                    },
                    "eol": {
                        "onshore": "treatment of platform, for onshore wind turbine",
                        "offshore": "treatment of platform, for offshore wind turbine",
                        "amount": target_component_masses["transition"] * -1,
                    },
                },
                "pile": {
                    "dataset": {
                        "onshore": "market for steel, low-alloyed, hot rolled",
                        "offshore": "market for steel, low-alloyed, hot rolled",
                        "amount": target_component_masses["pile"]
                    },
                    "eol": {
                        "onshore": "treatment of pile, for onshore wind turbine",
                        "offshore": "treatment of pile, for offshore wind turbine",
                        "amount": target_component_masses["pile"] * -1,
                    },
                },
                "medium-voltage transformer": {
                    "dataset": {
                        "onshore": "medium-voltage transformer production, for wind turbine",
                        "offshore": "medium-voltage transformer production, for wind turbine",
                        "amount": target_component_masses["medium-voltage transformer"]
                    },
                    "eol": {
                        "onshore": "treatment of medium-voltage transformer, for wind turbine",
                        "offshore": "treatment of medium-voltage transformer, for wind turbine",
                        "amount": target_component_masses["medium-voltage transformer"] * -1,
                    },
                },
                "high-voltage transformer": {
                    "dataset": {
                        "onshore": "high-voltage transformer production, for wind turbine",
                        "offshore": "high-voltage transformer production, for wind turbine",
                        "amount": target_component_masses["high-voltage transformer"]
                    },
                    "eol": {
                        "onshore": "treatment of high-voltage transformer, for wind turbine",
                        "offshore": "treatment of high-voltage transformer, for wind turbine",
                        "amount": target_component_masses["high-voltage transformer"] * -1,
                    },
                },
                "foundation": {
                    "dataset": {
                        "onshore": "foundation production, for onshore wind turbine",
                        "offshore": "foundation production, for offshore wind turbine",
                        "amount": target_component_masses["foundation"]
                    },
                    "eol": {
                        "onshore": "treatment of foundation, for onshore wind turbine",
                        "offshore": "treatment of foundation, for offshore wind turbine",
                        "amount": target_component_masses["foundation"] * -1,
                    },
                },
            }

            for component in [
                "rotor",
                "nacelle",
            ]:

                ds = ws.get_one(
                    self.database,
                    ws.equals("name", components_datasets[component]["dataset"][turbine_type]),
                )

                eol = ws.get_one(
                    self.database,
                    ws.equals("name", components_datasets[component]["eol"][turbine_type]),
                )

                moving["exchanges"].extend(
                    [
                        {
                            "amount": components_datasets[component]["dataset"]["amount"],
                            "type": "technosphere",
                            "unit": ds["unit"],
                            "name": ds["name"],
                            "product": ds["reference product"],
                            "location": ds["location"],
                            "uncertainty type": 0,
                        },
                        {
                            "amount": components_datasets[component]["eol"]["amount"],
                            "type": "technosphere",
                            "unit": eol["unit"],
                            "name": eol["name"],
                            "product": eol["reference product"],
                            "location": eol["location"],
                            "uncertainty type": 0,
                        },
                    ]
                )

                moving["comment"] += f"\n{component.capitalize()} mass: {components_datasets[component]['dataset']['amount'] / 1000} tons."

            moving["comment"] += f"\nRotor diameter: {target_component_masses['rotor diameter']} m. "
            moving["comment"] += f"\nHub height: {target_component_masses['hub height']} m. "
            moving["comment"] += f"\nNumber of turbines in park: {target_component_masses['turbines in park']}. "
            moving["comment"] += f"\nDistance to transformer: {target_component_masses['distance to transformer']} m. "
            if turbine_type == "offshore":
                moving["comment"] += f"\nSea depth: {target_component_masses['sea depth']} m. "
                moving["comment"] += f"\nPile height: {target_component_masses['pile height']} m. "
                moving["comment"] += f"\nDistance to coast: {target_component_masses['distance to coast']} m. "

            for component in [
                "tower",
                "grid connector",
                "foundation",
                "electronic cabinet",
                "transition",
                "medium-voltage transformer",
                "high-voltage transformer",
            ]:

                ds = ws.get_one(
                    self.database,
                    ws.equals("name", components_datasets[component]["dataset"][turbine_type]),
                )

                eol = ws.get_one(
                    self.database,
                    ws.equals("name", components_datasets[component]["eol"][turbine_type]),
                )

                fixed["exchanges"].extend(
                    [
                        {
                            "amount": components_datasets[component]["dataset"]["amount"],
                            "type": "technosphere",
                            "unit": ds["unit"],
                            "name": ds["name"],
                            "product": ds["reference product"],
                            "location": ds["location"],
                            "uncertainty type": 0,
                        },
                        {
                            "amount": components_datasets[component]["eol"]["amount"],
                            "type": "technosphere",
                            "unit": eol["unit"],
                            "name": eol["name"],
                            "product": eol["reference product"],
                            "location": eol["location"],
                            "uncertainty type": 0,
                        },
                    ]
                )

                if component in (
                        "electronic cabinet",
                        "medium-voltage transformer",
                        "high-voltage transformer",
                ):
                    fixed[
                        "comment"] += f"\n{component.capitalize()} mass: {components_datasets[component]['dataset']['amount']} units."
                else:
                    fixed["comment"] += f"\n{component.capitalize()} mass: {components_datasets[component]['dataset']['amount'] / 1000} tons."

            fixed["comment"] += f"\nRotor diameter: {target_component_masses['rotor diameter']} m. "
            fixed["comment"] += f"\nHub height: {target_component_masses['hub height']} m. "
            fixed["comment"] += f"\nNumber of turbines in park: {target_component_masses['turbines in park']}. "
            fixed["comment"] += f"\nDistance to transformer: {target_component_masses['distance to transformer']} m. "
            if turbine_type == "offshore":
                fixed["comment"] += f"\nSea depth: {target_component_masses['sea depth']} m. "
                fixed["comment"] += f"\nPile height: {target_component_masses['pile height']} m. "
                fixed["comment"] += f"\nDistance to coast: {target_component_masses['distance to coast']} m. "

            # add the new dataset to the database
            self.database.append(fixed)
            # add the new dataset to the database
            self.database.append(moving)

            for i, row in self.capacity_factors.iterrows():
                country = row["country"]
                cf = row[turbine_type]

                # check that cf is not NaN
                if np.isnan(cf):
                    continue

                production = int(get_electricity_production(
                    capacity_factor=cf,
                    power=int(power),
                    lifetime=20
                ))

                if turbine_type == "onshore":
                    dataset_name = "electricity production, wind, <1MW turbine, onshore"
                else:
                    dataset_name = "electricity production, wind, 1-3MW turbine, offshore"

                try:
                    electricity_ds = copy.deepcopy(ws.get_one(
                        self.database,
                        ws.equals("name", dataset_name),
                        ws.equals("location", country),
                    ))
                except ws.NoResults:
                    # fetch a Swiss dataset if we don't have one for the country
                    electricity_ds = copy.deepcopy(ws.get_one(
                        self.database,
                        ws.equals("name", dataset_name),
                        ws.equals("location", "DE"),
                    ))
                    electricity_ds = wurst.copy_to_new_location(electricity_ds, country)

                # modify the name of the dataset
                electricity_ds["name"] = f"electricity production, wind, {'{:.1f}'.format(int(power/1000))}MW turbine, {turbine_type}"
                electricity_ds["reference product"] = f"electricity, high voltage"
                electricity_ds["code"] = str(uuid.uuid4().hex)
                electricity_ds["comment"] = (
                    f"Generated for a {power} kW wind turbine."
                    f"Assumed lifetime: 20 years. "
                    f"Lifetime production: {production / 1e6} GWh. "
                    f"Capacity factor: {cf * 100}%."
                )

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
                self.add_to_index(electricity_ds)

                created_datasets.append(
                    (country, power)
                )

                results.append([country, cf, turbine_type, production, production/20])

        fleet_average_power = get_power_from_year(self.year, turbine_type)
        fleet_average_power = np.clip(
            fleet_average_power,
            0,
            max_power - 1000
        )
        lower_bound = 1  # MW
        upper_bound = 22  # MW

        # Generate the binned skewed distribution
        bin_edges, fleet_distribution = get_fleet_distribution(fleet_average_power, lower_bound, upper_bound, skew_factor=0.5)

        # create, for each country, a fleet average dataset for the wind turbines
        # considering the fleet capacity distribution

        wind_power_gen = get_wind_power_generation()

        for country in self.capacity_factors.loc[:, "country"].unique():
            if len([x for x in created_datasets if x[0] == country]) == 0:
                # no wind turbine dataset created for this country
                # e.g., offshore market for landlocked countries
                continue

            new_market_dataset = {
                "name": f"electricity production, wind, {turbine_type}",
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
                        "name": f"electricity production, wind, {'{:.1f}'.format(int(p/1000))}MW turbine, {turbine_type}",
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
