class MetalProperties:
    def __init__(self):
        # Define metal ions and their properties using a dictionary of dictionaries
        self.metal_properties = {
            "Fe": {
                "AtomicNumber": 26, 
                "IonicRadius": 74, 
                "Charge": 2, 
                "Electronegativity": 1.83, 
                "CoordinationNumber": 6, 
                "MolecularWeight": 55.85, 
                "IonizationEnergy": 762.5, 
                "ElectronAffinity": 15.7, 
                "HydrationEnergy": -1070, 
                "RedoxPotential": 0.77,
                "MeltingPoint": 1538,        # in Celsius
                "BoilingPoint": 2862,        # in Celsius
                "Density": 7.87,             # in g/cm^3
                "HeatCapacity": 0.449,       # in J/g·K
                "ThermalConductivity": 80.4, # in W/m·K
                "ElectricalConductivity": 10.04 # in 10^6 S/m
            },
            "Zn": {
                "AtomicNumber": 30, 
                "IonicRadius": 74, 
                "Charge": 2, 
                "Electronegativity": 1.65, 
                "CoordinationNumber": 4, 
                "MolecularWeight": 65.38, 
                "IonizationEnergy": 906.4, 
                "ElectronAffinity": -58, 
                "HydrationEnergy": -2046, 
                "RedoxPotential": -0.76,
                "MeltingPoint": 419.5,       # in Celsius
                "BoilingPoint": 907,         # in Celsius
                "Density": 7.14,             # in g/cm^3
                "HeatCapacity": 0.387,       # in J/g·K
                "ThermalConductivity": 116,  # in W/m·K
                "ElectricalConductivity": 16.6 # in 10^6 S/m
            },
            "Mg": {
                "AtomicNumber": 12, 
                "IonicRadius": 72, 
                "Charge": 2, 
                "Electronegativity": 1.31, 
                "CoordinationNumber": 6, 
                "MolecularWeight": 24.31, 
                "IonizationEnergy": 737.7, 
                "ElectronAffinity": -40, 
                "HydrationEnergy": -1900, 
                "RedoxPotential": -2.37,
                "MeltingPoint": 650,         # in Celsius
                "BoilingPoint": 1090,        # in Celsius
                "Density": 1.74,             # in g/cm^3
                "HeatCapacity": 1.023,       # in J/g·K
                "ThermalConductivity": 156,  # in W/m·K
                "ElectricalConductivity": 23.05 # in 10^6 S/m
            },
            "Mn": {
                "AtomicNumber": 25, 
                "IonicRadius": 83, 
                "Charge": 2, 
                "Electronegativity": 1.55, 
                "CoordinationNumber": 6, 
                "MolecularWeight": 54.94, 
                "IonizationEnergy": 717.3, 
                "ElectronAffinity": 0, 
                "HydrationEnergy": -1870, 
                "RedoxPotential": 1.51,
                "MeltingPoint": 1246,        # in Celsius
                "BoilingPoint": 2061,        # in Celsius
                "Density": 7.21,             # in g/cm^3
                "HeatCapacity": 0.479,       # in J/g·K
                "ThermalConductivity": 7.81, # in W/m·K
                "ElectricalConductivity": 6.3 # in 10^6 S/m
            },
            "Na": {
                "AtomicNumber": 11, 
                "IonicRadius": 102, 
                "Charge": 1, 
                "Electronegativity": 0.93, 
                "CoordinationNumber": 6, 
                "MolecularWeight": 22.99, 
                "IonizationEnergy": 495.8, 
                "ElectronAffinity": -52.8, 
                "HydrationEnergy": -406, 
                "RedoxPotential": -2.71,
                "MeltingPoint": 97.8,        # in Celsius
                "BoilingPoint": 883,         # in Celsius
                "Density": 0.97,             # in g/cm^3
                "HeatCapacity": 1.228,       # in J/g·K
                "ThermalConductivity": 142,  # in W/m·K
                "ElectricalConductivity": 21.0 # in 10^6 S/m
            },
            "Ca": {
                "AtomicNumber": 20, 
                "IonicRadius": 100, 
                "Charge": 2, 
                "Electronegativity": 1.00, 
                "CoordinationNumber": 6, 
                "MolecularWeight": 40.08, 
                "IonizationEnergy": 589.8, 
                "ElectronAffinity": -2.4, 
                "HydrationEnergy": -1576, 
                "RedoxPotential": -2.87,
                "MeltingPoint": 842,         # in Celsius
                "BoilingPoint": 1484,        # in Celsius
                "Density": 1.55,             # in g/cm^3
                "HeatCapacity": 0.647,       # in J/g·K
                "ThermalConductivity": 200,  # in W/m·K
                "ElectricalConductivity": 29.1 # in 10^6 S/m
            }
        }
        self.normalized_metal_data = self.normalize_metal_data()
    def normalize(self, value, min_value, max_value):
        if max_value - min_value == 0:
            return 0
        return (value - min_value) / (max_value - min_value)

    def normalize_metal_data(self):
        normalized_data = {}
        for metal, features in self.metal_properties.items():
            normalized_features = {}
            for feature, value in features.items():
                feature_values = [self.metal_properties[m][feature] for m in self.metal_properties]
                min_value = min(feature_values)
                max_value = max(feature_values)
                normalized_features[feature] = self.normalize(value, min_value, max_value)
            normalized_data[metal] = normalized_features
        return normalized_data

    def get_normalized_metal_features(self, metal_type):
        return self.normalized_metal_data.get(metal_type, {})